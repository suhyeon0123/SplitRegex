import torch

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'SoftConciseNormalForm' )))

from collections import Counter
import submodels.SoftConciseNormalForm.synthesizer_snort
import submodels.SoftConciseNormalForm.synthesizer

from seq2seq.dataset.dataset import Vocabulary
from submodels.SoftConciseNormalForm.examples import Examples
from rpni import synthesis as rpni_synthesis

def split(strings, label, no_split=False):
    vocab = Vocabulary()

    batch = []

    if no_split:
        for batch_idx in range(len(strings)):
            set = []
            for set_idx in range(10):
                seq = []
                seq.append(''.join(map(lambda x:vocab.itos[x], strings[batch_idx, set_idx][strings[batch_idx, set_idx] != strings.max()].tolist())))
                set.append(seq)
            batch.append(set)
        return batch


    label = [i.tolist() for i in label]
    tmp = torch.LongTensor(label).transpose(0, 1).squeeze(-1).tolist()
    predict_dict = [dict(Counter(l)) for l in tmp]

    # print(torch.tensor(label))
    # print(torch.tensor(label).size)
    # split_size = torch.tensor(label)[torch.tensor(label) != 10].max().item() + 1
    split_size = torch.tensor(label)[torch.tensor(label) != 63].max().item() + 1
    for batch_idx in range(len(strings)):
        set = []
        for set_idx in range(10):

            src_seq = strings[batch_idx, set_idx].tolist()  # list of 10 alphabet
            predict_seq_dict = predict_dict[batch_idx * 10 + set_idx]  # predict label. ex. {0.0: 2, 1.0: 1, 11.0: 7}
            seq = []
            idx = 0
            for seq_id in range(split_size):
                tmp = ''
                if seq_id in predict_seq_dict.keys():
                    for _ in range(predict_seq_dict[float(seq_id)]):
                        tmp += vocab.itos[src_seq[idx]]
                        idx += 1
                seq.append(tmp)
            set.append(seq)
        batch.append(set)

    return batch

def generate_split_regex(splited_pos, splited_neg, split_model=False, count_limit=1000, alphabet_size=5, data_type='random'):
    regex = []

    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    splited_pos = list(filter(lambda x:any(x),splited_pos))
    splited_neg = list(filter(lambda x: any(x), splited_neg))

    for sub_id in range(split_size):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
        for set_idx in range(len(splited_neg)):
            neg.append(splited_neg[set_idx][0])
        if not neg:
            neg.append('')
            #if len(splited_neg[set_idx]) > sub_id:
            # try:
            #     neg.append(splited_neg[set_idx][0])
            # except:
            #     neg.append('')
            # #else:
            # #    neg.append('')

        sub_pos_set = set(pos)
        sub_neg_set = set(neg)

        if sub_id + 1 == split_size:
            prefix = ''.join(regex)
        else:
            sub_neg_set -= sub_pos_set
            prefix = None

        # sub_neg_set -= sub_pos_set
        # if sub_id != 0:
        #     prefix = ''.join(regex)
        # else:
        #     prefix = None

        #
        # if '' in sub_pos_set:
        #     sub_pos_set.remove('')
        #     sub_pos_set.add('@epsilon')
        #
        # if '' in sub_neg_set:
        #     sub_neg_set.remove('')
        #     sub_neg_set.add('@epsilon')

        print('Splited Positive Strings:', sub_pos_set)
        print('Splited Negative Strings:', sub_neg_set)

        if data_type == 'practical':
            if split_model and sub_id == 0:
                count_limit = int(count_limit / 8)
            elif split_model and sub_id == split_size-1:
                count_limit = int(count_limit * 4)



        if data_type == 'random':
            tmp = submodels.SoftConciseNormalForm.synthesizer.synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model, prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
        else:
            tmp, candidate = submodels.SoftConciseNormalForm.synthesizer_snort.synthesis(
                Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model,
                prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)

        if tmp is None:
            if split_model and data_type == 'practical':
                tmp = candidate
            else:
                return None, 0

        # tmp = rpni_synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model, prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
        # if tmp is None:
        #     return None, 0
        #
        regex.append('(' + repr(tmp) + ')')

    return ''.join(regex), split_size







