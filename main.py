import argparse
import time
import torch
import os, sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'SoftConsiceNormalFrom' )))

#from util import *
#from examples import *

from seq2seq.dataset import pos_neg_dataset
from seq2seq.util.checkpoint import Checkpoint
from split import split, generate_split_regex

from submodels.SoftConciseNormalForm.examples import Examples
from submodels.SoftConciseNormalForm.util import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/pos_neg_5.csv', dest='data_path',
                    help='Path to data')
parser.add_argument('--batch_size', action='store', dest='batch_size',
                    help='batch size', default=1)
parser.add_argument('--checkpoint_pos', default='./saved_models/hidden_512/best_accuracy', dest='checkpoint_pos',
                    help='path to checkpoint for splitting positive strings ')
parser.add_argument('--checkpoint_neg', default='./saved_models/hidden_512/best_accuracy', dest='checkpoint_neg',
                    help='path to checkpoint for splitting negative strings ')
parser.add_argument('--sub_model', action='store', dest='sub_model', default='set2regex',
                    help='sub model used in generating sub regex from sub strings')

opt = parser.parse_args()





def print_tensor_set(tensor_set):
    output_strings = []
    for i in range(tensor_set.shape[0]):
        output_strings.append(''.join(map(str, tensor_set[i][tensor_set[i] != tensor_set.max()].tolist())))

    return output_strings


def main():
    data = pos_neg_dataset.get_loader(opt.data_path, batch_size=opt.batch_size, shuffle=True)

    pos_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_pos))
    neg_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_neg))

    pos_split_model = pos_checkpoint.model
    neg_split_model = neg_checkpoint.model

    pos_split_model.eval()
    neg_split_model.eval()


    dc_time_total = 0
    direct_time_total = 0

    dc_correct_count = 0
    dc_correct = False
    direct_correct = False

    dc_win = 0
    direct_win = 0


    count_limit = 2000


    for count, (pos, neg, regex) in enumerate(data):
        pos, neg, regex = pos_neg_dataset.batch_preprocess(pos, neg, regex)

        pos_set = print_tensor_set(pos[0])
        neg_set = print_tensor_set(neg[0])

        print('-'*50)
        print('Positive Strings:', ', '.join(pos_set))
        print('Negative Strings:', ', '.join(neg_set))
        print('Target Regex:', ''.join(regex[0]))
        print('-'*50)

        # via DC
        start_time = time.time()

        _, _, other = pos_split_model(pos, None, regex)
        splited_pos = split(pos, other['sequence'])  # batch, set, seq

        _, _, other = neg_split_model(neg)
        splited_neg = split(neg, other['sequence'])  # batch, set, seq

        batch_predict = []
        for batch_idx in range(len(pos)):
            result, split_size = generate_split_regex(splited_pos[batch_idx], splited_neg[batch_idx], neg_set, True, count_limit)
            batch_predict.append(result)

        end_time = time.time()

        dc_time_taken = end_time - start_time
        dc_time_total += dc_time_taken

        if batch_predict[0] is not None:
            dc_correct = is_solution(batch_predict[0], Examples(pos=pos_set, neg=neg_set), membership)
        else:
            dc_correct = False

        if dc_correct:
            dc_correct_count += 1

        print(f'{count}th Generated Regex (via DC): {batch_predict[0]} ({dc_correct}), Time Taken: ', end_time - start_time)

        # direct
        start_time = time.time()

        _, _, other = pos_split_model(pos, None, regex)
        splited_pos = split(pos, other['sequence'], no_split=True)  # batch, set, seq

        _, _, other = neg_split_model(neg)
        splited_neg = split(neg, other['sequence'], no_split=True)  # batch, set, seq


        batch_predict = []
        for batch_idx in range(len(pos)):
            result, split_size = generate_split_regex(splited_pos[batch_idx], splited_neg[batch_idx], neg_set, False, count_limit)
            batch_predict.append(result)

        end_time = time.time()

        direct_time_taken = end_time - start_time
        direct_time_total += direct_time_taken

        if batch_predict[0] is not None:
            direct_correct = True
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

        print(f'{count}th Generated Regex (direct): {batch_predict[0]}, Time Taken: ', direct_time_taken)
        print(f'Divide-and-conquer win rate over Direct: {dc_win / (dc_win + direct_win + 1e-9) * 100:.4f}%, Direct Total Time: {direct_time_total:.4f}, DC Total Time: {dc_time_total:.4f}')
        print('-'*50)


if __name__ == "__main__":
    main()
