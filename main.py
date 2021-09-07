import argparse
import time
import pickle
import seq2seq.dataset.dataset as dataset
from seq2seq.dataset.dataset import Vocabulary
from seq2seq.util.checkpoint import Checkpoint
from split import split, generate_split_regex

from submodels.SoftConciseNormalForm.examples import Examples
from submodels.SoftConciseNormalForm.util import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/practical_data/test_practicalregex.csv', dest='data_path',
                    help='Path to data')
parser.add_argument('--log_path', default='./log_data/practical', dest='log_path',
                    help='Path to save log data')
parser.add_argument('--batch_size', action='store', dest='batch_size',
                    help='batch size', default=1)
parser.add_argument('--checkpoint_pos', default='./saved_models/practical2/rnntype_gru_hidden_128/best_accuracy', dest='checkpoint_pos',
                    help='path to checkpoint for splitting positive strings ')
parser.add_argument('--checkpoint_neg', default='./saved_models/rnntype_gru_hidden_128/best_accuracy', dest='checkpoint_neg',
                    help='path to checkpoint for splitting negative strings ')
parser.add_argument('--sub_model', action='store', dest='sub_model', default='alpharegex',
                    help='sub model used in generating sub regex from sub strings')
parser.add_argument('--data_type', action='store', dest='data_type', default='practical',
                    help='data type: random or practical')
parser.add_argument('--alphabet_size', action='store', dest='alphabet_size',
                    help='define the alphabet size of the regex', type=int, default=8)



opt = parser.parse_args()

def print_tensor_set(tensor_set):
    output_strings = []
    vocab = Vocabulary()
    for i in range(tensor_set.shape[0]):
        output_strings.append(''.join(map(lambda x: vocab.itos[x], tensor_set[i][tensor_set[i] != vocab.stoi['<pad>']].tolist())))

    output_strings = list(filter(None, output_strings))
    return output_strings


def main():

    if 'random' in opt.data_type:
        MAX_SEQUENCE_LENGTH = 10
    else:
        MAX_SEQUENCE_LENGTH = 15

    data = dataset.get_loader(opt.data_path, batch_size=opt.batch_size, object='test', shuffle=True, max_len=MAX_SEQUENCE_LENGTH)

    pos_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_pos))
    neg_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_neg))

    pos_split_model = pos_checkpoint.model
    neg_split_model = neg_checkpoint.model

    pos_split_model.eval()
    neg_split_model.eval()


    dc_time_total = 0
    direct_time_total = 0

    dc_correct_count = 0
    direct_correct_count = 0

    MAX_TIME_LIMIT = 20
    COUNT_LIMIT = 5000

    dc_win = 0
    direct_win = 0


    for count, tuple in enumerate(data):
        pos, neg, tmp_regex, valid_pos, valid_neg = tuple
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


        # via DC
        start_time = time.time()

        _, _, other = pos_split_model(pos, None, regex)
        splited_pos, sigma_lst = split(pos, other['sequence'])  # batch, set, seq

        # _, _, other = neg_split_model(neg)
        splited_neg, _ = split(neg, other['sequence'], no_split=True)  # batch, set, seq

        batch_predict = []
        for batch_idx in range(len(pos)):
            result, split_size = generate_split_regex(splited_pos[batch_idx], splited_neg[batch_idx], True,  COUNT_LIMIT, alphabet_size=opt.alphabet_size, data_type=opt.data_type, sigma_lst=sigma_lst)
            batch_predict.append(result)

        end_time = time.time()


        dc_time_taken = end_time - start_time
        timeout = False
        if dc_time_taken > MAX_TIME_LIMIT:
            dc_time_taken = MAX_TIME_LIMIT
            timeout = True
        dc_time_total += dc_time_taken

        dc_answer = batch_predict[0]

        if dc_answer is not None and not timeout:
            dc_correct = is_solution(dc_answer, Examples(pos=pos_set, neg=neg_set), membership)
        else:
            dc_correct = False

        if dc_correct:
            dc_correct_count += 1

        print(f'{count}th Generated Regex (via DC): {dc_answer} ({dc_correct}), Time Taken: ', dc_time_taken)


        # direct
        start_time = time.time()

        #_, _, other = pos_split_model(pos, None, regex)
        splited_pos, _ = split(pos, other['sequence'], no_split=True)  # batch, set, seq

        #_, _, other = neg_split_model(neg)
        splited_neg, _ = split(neg, other['sequence'], no_split=True)  # batch, set, seq


        batch_predict = []
        for batch_idx in range(len(pos)):
            result, split_size = generate_split_regex(splited_pos[batch_idx], splited_neg[batch_idx], False, COUNT_LIMIT, alphabet_size=opt.alphabet_size, data_type=opt.data_type)
            batch_predict.append(result)

        end_time = time.time()


        direct_time_taken = end_time - start_time
        timeout = False
        if direct_time_taken > MAX_TIME_LIMIT:
            direct_time_taken = MAX_TIME_LIMIT
            timeout = True
        direct_time_total += direct_time_taken

        direct_answer = batch_predict[0]
        if direct_answer is not None and not timeout:
            direct_correct = True
            direct_correct_count += 1
        else:
            direct_correct = False


        if dc_correct:
            if direct_correct:
                if direct_time_taken > dc_time_taken:
                    dc_win += 1
                else:
                    direct_win += 1
            else:
                dc_win += 1
        elif direct_correct:
            direct_win += 1


        print(f'{count}th Generated Regex (direct): {direct_answer}, Time Taken: ', direct_time_taken)
        print(f'Divide-and-conquer win rate over Direct: {dc_win / (dc_win + direct_win + 1e-9) * 100:.4f}%, Direct Total Time: {direct_time_total:.4f}, DC Total Time: {dc_time_total:.4f}')
        print(f'DC Success Ratio: {dc_correct_count / (count + 1) * 100:.4f}%, Direct Success Ratio: {direct_correct_count / (count + 1) * 100:.4f}%')
        print('-'*50)


        log_data = dict()
        log_data['Target_string'] = ''.join(regex[0])
        log_data['pos'] = pos_set
        log_data['neg'] = neg_set
        log_data['pos_validation'] = valid_pos_set
        log_data['neg_validation'] = valid_neg_set
        log_data['DC_answer'] = dc_answer
        log_data['Direct_answer'] = direct_answer
        log_data['win_rate'] = dc_win / (dc_win + direct_win + 1e-9) * 100
        log_data['DC_success_ratio'] = dc_correct_count / (count + 1) * 100
        log_data['Direct_success_ratio'] = direct_correct_count / (count + 1) * 100
        log_data['DC_time'] = dc_time_taken
        log_data['Direct_time'] = direct_time_taken
        log_data['DC_total_time'] = dc_time_total
        log_data['Direct_total_time'] = direct_time_total

        with open(opt.log_path + '/' + str(count) + '.pickle', 'wb') as fw:
            pickle.dump(log_data, fw)

if __name__ == "__main__":
    main()
