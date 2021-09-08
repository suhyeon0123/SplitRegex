import torch

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'SoftConciseNormalForm')))

from collections import Counter
import submodels.SoftConciseNormalForm.synthesizer
from submodels.SoftConciseNormalForm.parsetree import *

import submodels.SoftConciseNormalForm.synthesizer_snort
from submodels.SoftConciseNormalForm.util_snort import *

from seq2seq.dataset.dataset import Vocabulary
from submodels.SoftConciseNormalForm.examples import Examples
from rpni import synthesis as rpni_synthesis


def is_last_sigma(lst, split_size):

    try:
        idx = len(lst) - 1 - list(reversed(lst)).index(split_size)
    except:
        return False

    if idx != 9 and lst[idx+1] == 0:
        return True


def split(strings, label, no_split=False):
    vocab = Vocabulary()

    splited_string = []

    if no_split:
        for batch_idx in range(len(strings)):
            set = []
            for set_idx in range(10):
                seq = []
                seq.append(''.join(map(lambda x: vocab.itos[x], strings[batch_idx, set_idx][
                    strings[batch_idx, set_idx] != strings.max()].tolist())))
                set.append(seq)
            splited_string.append(set)

        return splited_string, None

    label = [i.tolist() for i in label]
    tmp = torch.LongTensor(label).transpose(0, 1).squeeze(-1).tolist()

    split_size = torch.tensor(label)[torch.tensor(label) != vocab.stoi['<pad>']].max().item()
    if any(map(lambda x:is_last_sigma(x, split_size),tmp)):
        split_size += 1

    label2 = []
    sigma_lst = []
    for templete in tmp:
        tmp2 = []
        sigma_lst2 = []
        now = 0

        for element in templete:
            if element != 0:
                if now != element and element != vocab.stoi['<pad>']:
                    for _ in range(element - len(sigma_lst2)):
                        sigma_lst2.append(False)
                tmp2.append(element)
                now = element
            else:
                if not sigma_lst2 or not sigma_lst2[-1]:
                    sigma_lst2.append(True)
                tmp2.append(now + 1)

        while len(sigma_lst2) < split_size:
            sigma_lst2.append(False)

        label2.append(tmp2)
        sigma_lst.append(sigma_lst2)



    predict_dict = [dict(Counter(l)) for l in label2]
    print(predict_dict)
    for batch_idx in range(len(strings)):
        set = []
        for set_idx in range(10):

            src_seq = strings[batch_idx, set_idx].tolist()  # list of 30 alphabet
            predict_seq_dict = predict_dict[batch_idx * 10 + set_idx]  # predict label. ex. {0.0: 2, 1.0: 1, 11.0: 7}
            seq = []
            idx = 0
            for seq_id in range(1, split_size + 1):
                tmp = ''
                if seq_id in predict_seq_dict.keys():
                    for _ in range(predict_seq_dict[float(seq_id)]):
                        tmp += vocab.itos[src_seq[idx]]
                        idx += 1
                seq.append(tmp)
            set.append(seq)
        splited_string.append(set)

    return splited_string, sigma_lst

def is_satisfy_pos(regex, examples, membership):
    for string in examples.getPos():
        if not membership(regex, string):
            return False
    return True

def get_sigma(examples):
    if is_satisfy_pos('\d*', examples, membership):
        return r"\d*"
    elif is_satisfy_pos('\w*', examples, membership):
        return r"\w*"
    else:
        return r".*"



def generate_split_regex(splited_pos, splited_neg, split_model=False, count_limit=1000, alphabet_size=5,
                         data_type='random', sigma_lst=None, submodel='alpharegex'):
    regex = []

    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    splited_pos = list(filter(lambda x: any(x), splited_pos))
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

        sub_pos_set = set(pos)
        sub_neg_set = set(neg)


        if sub_id + 1 == split_size:
            prefix = ''.join(regex)
        else:
            sub_neg_set -= sub_pos_set
            prefix = None

        print('Splited Positive Strings:', sub_pos_set)
        print('Splited Negative Strings:', sub_neg_set)

        if len(sub_pos_set) == 1:
            regex.append('(' + sub_pos_set.pop() + ')')

        if submodel == 'blue_finge':
            tmp = rpni_synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model, prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
        elif submodel == 'alpharegex':
            if data_type == 'random':
                if sigma_lst is not None and sub_id + 1 != split_size and any(list(map(lambda x: x[sub_id], sigma_lst))):
                    tmp = repr(KleenStar(Or(*[Character(str(x)) for x in range(alphabet_size)])))
                else:
                    tmp = repr(submodels.SoftConciseNormalForm.synthesizer.synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set),
                                                                                count_limit,
                                                                                start_with_no_concat=split_model,
                                                                                prefix_for_neg_test=prefix,
                                                                                suffix_for_neg_test=None,
                                                                                 alphabet_size=alphabet_size))
            else:
                if sigma_lst is not None and sub_id + 1 != split_size and any(list(map(lambda x: x[sub_id], sigma_lst))):
                    tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
                else:
                    tmp, _ = submodels.SoftConciseNormalForm.synthesizer_snort.synthesis(
                        Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model,
                        prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
                    tmp = repr(tmp)
        elif submodel == 'set2regex':
            pass
        elif submodel == 'regexgenerator':
            pass

        if tmp == 'None':
            return None, 0


        regex.append('(' + tmp + ')')

    return ''.join(regex), split_size
