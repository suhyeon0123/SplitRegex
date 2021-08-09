import torch

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'SoftConciseNormalForm' )))

from collections import Counter
from submodels.SoftConciseNormalForm.synthesizer import synthesis
from submodels.SoftConciseNormalForm.examples import Examples

def split(strings, label, no_split=False):

    batch = []

    if no_split:
        for batch_idx in range(len(strings)):
            set = []
            for set_idx in range(10):
                seq = []
                seq.append(''.join(map(str, strings[batch_idx, set_idx][strings[batch_idx, set_idx] != strings.max()].tolist())))
                set.append(seq)
            batch.append(set)
        return batch


    label = [i.tolist() for i in label]
    tmp = torch.LongTensor(label).transpose(0, 1).squeeze(-1).tolist()
    predict_dict = [dict(Counter(l)) for l in tmp]

    split_size = torch.tensor(label)[torch.tensor(label) != 10].max().item() + 1

    for batch_idx in range(len(strings)):
        set = []
        for set_idx in range(10):

            src_seq = strings[batch_idx, set_idx].tolist()  # list of 10 alphabet
            #print(src_seq)
            predict_seq_dict = predict_dict[batch_idx * 10 + set_idx]  # predict label. ex. {0.0: 2, 1.0: 1, 11.0: 7}
            seq = []
            idx = 0
            for seq_id in range(split_size):
                tmp = ''
                if seq_id in predict_seq_dict.keys():
                    for _ in range(predict_seq_dict[float(seq_id)]):
                        tmp += str(src_seq[idx])
                        idx += 1
                seq.append(tmp)
            #print(seq)
            set.append(seq)
        batch.append(set)

    return batch

def generate_split_regex(splited_pos, splited_neg, split_model=False, count_limit=1000):
    regex = []

    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    for sub_id in range(split_size):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
            #if len(splited_neg[set_idx]) > sub_id:
            neg.append(splited_neg[set_idx][0])
            #else:
            #    neg.append('')

        sub_pos_set = set(pos)
        sub_neg_set = set(neg)
        #neg_set = set(neg_set)

        sub_neg_set -= sub_pos_set
        #neg_set -= sub_pos_set

        print('Splited Positive Strings:', sub_pos_set)
        print('Splited Negative Strings:', sub_neg_set)

        if sub_id + 1 == split_size:
            prefix = ''.join(regex)
        else:
            prefix = None

        tmp = synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model, prefix_for_neg_test=prefix, suffix_for_neg_test=None)

        if tmp is None:
            return None, 0

        print(tmp)
        regex.append('(' + tmp + ')')

    return ''.join(regex), split_size







