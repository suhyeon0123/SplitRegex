import argparse
import time
import pickle
import seq2seq.dataset.dataset as dataset
from seq2seq.dataset.dataset import Vocabulary
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.seed import seed_all
from seq2seq.models import *
from split import *
import signal
import configparser
import pathlib
import torch
import numpy as np

from submodels.SCNF.examples import Examples
from submodels.SCNF.util import *
import FAdo.reex as reex

from multiprocessing import Process, Manager


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/practical_data/test_practicalregex.csv', dest='data_path',
                    help='Path to data')
parser.add_argument('--log_path', default='./log_data/practical', dest='log_path',
                    help='Path to save log data')
parser.add_argument('--batch_size', action='store', dest='batch_size',
                    help='batch size', default=1)
parser.add_argument('--checkpoint_pos', default='./saved_models/practical/rnntype_gru_hidden_128/best_accuracy', dest='checkpoint_pos',
                    help='path to checkpoint for splitting positive strings ')
parser.add_argument('--sub_model', action='store', dest='sub_model', default='alpharegex',
                    help='sub model used in generating sub regex from sub strings')
parser.add_argument('--data_type', action='store', dest='data_type', default='practical',
                    help='data type: random or practical')
parser.add_argument('--alphabet_size', action='store', dest='alphabet_size',
                    help='define the alphabet size of the regex', type=int, default=10)
parser.add_argument('--time_limit', action='store', dest='time_limit',
                    help='time_limit', type=int, default=3)
parser.add_argument('--synthesis_strategy', action='store', dest='synthesis_strategy', default='sequential_basic',
                    help='synthesis_strategy: sequential_prefix, parallel, sequential_basic')
parser.add_argument('--exclude_GT', action='store_true', dest='exclude_GT', help='decide to not infer GT split')     
parser.add_argument('--exclude_Direct', action='store_true', dest='exclude_Direct', help='decide to not infer Direct split')     




class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeOutException()


def membership(regex, string):
    return bool(re.fullmatch(regex, string))

def membership2(regex, string):
    return reex.str2regexp(regex).evalWordP(string)


opt = parser.parse_args()

def print_tensor_set(tensor_set):
    output_strings = []
    vocab = Vocabulary()
    for i in range(tensor_set.shape[0]):
        output_strings.append(''.join(map(lambda x: vocab.itos[x], tensor_set[i][tensor_set[i] != vocab.stoi['<pad>']].tolist())))

    output_strings = list(filter(None, output_strings))
    return output_strings


def main():
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    seed_all(int(config['seed']['main']))

    COUNT_LIMIT = 1000000

    if 'regex_generator' in opt.sub_model:
        MAX_TIME_LIMIT = 15
    else:
        MAX_TIME_LIMIT = opt.time_limit
    
    if 'random' in opt.data_type:
        MAX_SEQUENCE_LENGTH = 10
    else:
        MAX_SEQUENCE_LENGTH = 15

    if 'blue_fringe' in opt.sub_model:
        membership_type = membership2
    else:
        membership_type = membership

    use_prefix_every = False
    if opt.synthesis_strategy == 'sequential_basic':
        generate_regex_from_split = generate_split_regex_sequential
    elif opt.synthesis_strategy == 'parallel':
        generate_regex_from_split = generate_split_regex_in_parallel
    elif opt.synthesis_strategy == 'sequential_prefix':
        generate_regex_from_split = generate_split_regex_sequential
        use_prefix_every = True
    else:
        raise Exception('unknown synthesis strategy')


    data = dataset.get_loader(opt.data_path, batch_size=opt.batch_size, object='test', shuffle=True, max_len=MAX_SEQUENCE_LENGTH)

    pos_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_pos))
    pos_split_model = pos_checkpoint.model
    pos_split_model.eval()

    direct_time_total, dc_time_total, gt_time_total = 0, 0, 0
    direct_correct_count, dc_correct_count, gt_correct_count = 0, 0, 0
    direct_win, dc_win = 0, 0

    manager = Manager()
    return_dict = manager.dict()


    for count, tuple in enumerate(data):
        
        pos, neg, tmp_regex, valid_pos, valid_neg, tag = tuple

        # blue_fringe cannot handle special character '_' and '!'
        if opt.sub_model == 'blue_fringe' and opt.data_type == 'practical':
            pos = list(map(lambda x: list(map(lambda y:torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y, x)), pos))
            neg = list(map(lambda x: list(map(lambda y: torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y, x)), neg))
            valid_pos = list(map(lambda x: list(map(lambda y: torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y, x)), valid_pos))
            valid_neg = list(map(lambda x: list(map(lambda y: torch.tensor([61]) if y.item() == 62 or y.item() == 63 else y, x)), valid_neg))
            

        pos, neg, regex = dataset.batch_preprocess(pos, neg, tmp_regex)
        valid_pos, valid_neg, regex = dataset.batch_preprocess(valid_pos, valid_neg, tmp_regex)
        
        pos_set = print_tensor_set(pos[0])
        neg_set = print_tensor_set(neg[0])
        valid_pos_set = print_tensor_set(valid_pos[0])
        valid_neg_set = print_tensor_set(valid_neg[0])

        if ', '.join(pos_set) == '':
            continue

        print('-'*50)
        print('Positive Strings:', ', '.join(pos_set))
        print('Negative Strings:', ', '.join(neg_set))
        print('Target Regex:', ''.join(regex[0]))
        print('-'*50)


        # via divide and conquer ---------------------------
        start_time = time.time()
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(MAX_TIME_LIMIT)


        try:
            _, _, other = pos_split_model(pos, None, regex)
            splited_pos, sigma_lst = split(pos, other['sequence'])  # batch, set, seq

            # _, _, other = neg_split_model(neg)
            splited_neg, _ = split(neg, other['sequence'], no_split=True)  # batch, set, seq

            dc_answer, split_size = generate_regex_from_split(splited_pos[0], splited_neg[0], True,  COUNT_LIMIT, alphabet_size=opt.alphabet_size, data_type=opt.data_type, sigma_lst=sigma_lst, submodel=opt.sub_model, return_dict=return_dict, use_prefix_every=use_prefix_every)
        except TimeOutException as e:
            print(e)
            print('time limit')
            dc_answer = None
        end_time = time.time()
        signal.alarm(0)
        if dc_answer is None:
            dc_correct = False
        else:
            try:
                dc_correct = is_solution(dc_answer, Examples(pos=pos_set, neg=neg_set), membership_type)
            except:
                dc_correct = False


        if dc_correct:
            dc_correct_count += 1
            dc_time_taken = end_time - start_time
        else:
            dc_time_taken = MAX_TIME_LIMIT
        dc_time_total += dc_time_taken


        print(f'{count}th Generated Regex (via DC): {dc_answer} ({dc_correct}), Time Taken: ', dc_time_taken)



        # via direct -------------------------------------
        if not opt.exclude_Direct:

            start_time = time.time()
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(MAX_TIME_LIMIT)

            try:
                #_, _, other = pos_split_model(pos, None, regex)
                splited_pos, _ = split(pos, other['sequence'], no_split=True)  # batch, set, seq

                #_, _, other = neg_split_model(neg)
                splited_neg, _ = split(neg, other['sequence'], no_split=True)  # batch, set, seq

                direct_answer, split_size = generate_split_regex_sequential(splited_pos[0], splited_neg[0], False, COUNT_LIMIT, alphabet_size=opt.alphabet_size, data_type=opt.data_type, submodel=opt.sub_model, return_dict=return_dict, use_prefix_every=use_prefix_every)
            except Exception as e:
                print('time limit')
                direct_answer = None
            end_time = time.time()
            signal.alarm(0)

            if direct_answer is None:
                direct_correct = False
            else:
                try:
                    direct_correct = is_solution(direct_answer, Examples(pos=pos_set, neg=neg_set), membership_type)
                except:
                    direct_correct = False
            
            if direct_correct:
                direct_correct_count += 1
                direct_time_taken = end_time - start_time
            else:
                direct_time_taken = MAX_TIME_LIMIT
            direct_time_total += direct_time_taken

            print(f'{count}th Generated Regex (direct): {direct_answer}, Time Taken: ', direct_time_taken)


        # via ground truth -----------------------------------------------------------------
        if not opt.exclude_GT:
            start_time = time.time()
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(MAX_TIME_LIMIT)


            try:
                gt_answer = None

                _, _, other = pos_split_model(pos, None, regex)
                splited_pos, sigma_lst = split(pos, np.array(tag, dtype="object").T)  # batch, set, seq

                # _, _, other = neg_split_model(neg)
                splited_neg, _ = split(neg, other['sequence'], no_split=True)  # batch, set, seq

                gt_split_size = len(splited_pos[0][0])

                gt_answer, split_size = generate_regex_from_split(splited_pos[0], splited_neg[0], True,  COUNT_LIMIT, alphabet_size=opt.alphabet_size, data_type=opt.data_type, sigma_lst=sigma_lst, submodel=opt.sub_model, return_dict=return_dict, use_prefix_every=use_prefix_every)
            except Exception as e:
                print(e)
                print('time limit')
            end_time = time.time()
            signal.alarm(0)

            if gt_answer is None:
                gt_correct = False
            else:
                try:
                    gt_correct = is_solution(gt_answer, Examples(pos=pos_set, neg=neg_set), membership_type)
                except:
                    gt_correct = False  
                
            if gt_correct:
                gt_correct_count += 1
                gt_time_taken = end_time - start_time
            else:
                gt_time_taken = MAX_TIME_LIMIT
            gt_time_total += gt_time_taken

            print(f'{count}th Generated Regex (via GT): {gt_answer} ({gt_correct}), Time Taken: ', gt_time_taken)


        # win rate
        if not opt.exclude_Direct:
            if dc_correct and direct_correct:
                if direct_time_taken > dc_time_taken:
                    dc_win += 1
                else:
                    direct_win += 1
            elif dc_correct:
                dc_win += 1
            elif direct_correct:
                direct_win += 1

        if not opt.exclude_Direct:
            print(f'Divide-and-conquer win rate over Direct: {dc_win / (dc_win + direct_win + 1e-9) * 100:.4f}%, Direct Total Time: {direct_time_total:.4f}, DC Total Time: {dc_time_total:.4f}')
            print(f'DC Success Ratio: {dc_correct_count / (count + 1) * 100:.4f}%, Direct Success Ratio: {direct_correct_count / (count + 1) * 100:.4f}%')
            print('-'*50)
        else:
            print(f'DC Total Time: {dc_time_total:.4f}')
            print(f'DC Success Ratio: {dc_correct_count / (count + 1) * 100:.4f}%')
            print('-'*50)


        log_data = dict()
        log_data['Target_string'] = ''.join(regex[0])
        log_data['pos'] = pos_set
        log_data['neg'] = neg_set
        log_data['pos_validation'] = valid_pos_set
        log_data['neg_validation'] = valid_neg_set
        log_data['DC_answer'] = dc_answer
        log_data['DC_success_ratio'] = dc_correct_count / (count + 1) * 100
        log_data['DC_time'] = dc_time_taken
        log_data['DC_total_time'] = dc_time_total

        if not opt.exclude_Direct:
            log_data['Direct_answer'] = direct_answer
            log_data['win_rate'] = dc_win / (dc_win + direct_win + 1e-9) * 100
            log_data['Direct_success_ratio'] = direct_correct_count / (count + 1) * 100
            log_data['Direct_time'] = direct_time_taken
            log_data['Direct_total_time'] = direct_time_total

        if not opt.exclude_GT:
            log_data['GT_answer'] = gt_answer
            log_data['GT_success_ratio'] = gt_correct_count / (count + 1) * 100
            log_data['GT_time'] = gt_time_taken
            log_data['GT_total_time'] = gt_time_total



        log_path = opt.log_path + '/{}'.format(opt.sub_model)

        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        with open(log_path + '/' + str(count) + '.pickle', 'wb') as fw:
            pickle.dump(log_data, fw)

if __name__ == "__main__":
    main()
