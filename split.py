import torch

import os, sys
from multiprocessing import Process, Manager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'SoftConciseNormalForm')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'RegexGenerator')))


from collections import Counter
from submodels.RegexGenerator.batch import *
import submodels.SoftConciseNormalForm.synthesizer
from submodels.SoftConciseNormalForm.parsetree import *

import submodels.SoftConciseNormalForm.synthesizer_snort
from submodels.SoftConciseNormalForm.util_snort import *

from seq2seq.dataset.dataset import Vocabulary
from submodels.SoftConciseNormalForm.examples import Examples
from rpni import synthesis as rpni_synthesis


class Ex():
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

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
        #if sub_id != 0:
            prefix = ''.join(regex)
        else:
            sub_neg_set -= sub_pos_set
            prefix = None



        print('Splited Positive Strings:', sub_pos_set)
        print('Splited Negative Strings:', sub_neg_set)

        if len(sub_pos_set) == 1:
            regex.append('(' + sub_pos_set.pop() + ')')
            continue

        if submodel == 'blue_fringe':
            count_limit = 1000000000
            tmp = rpni_synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model, prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
            tmp = str(tmp)
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
        elif submodel == 'regex_generator':
            if sigma_lst is not None and sub_id + 1 != split_size and any(list(map(lambda x: x[sub_id], sigma_lst))):
                tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
            else:
                tmp = execute([Ex(list(sub_pos_set), list(sub_neg_set))]).replace('++', '+')

        if tmp == 'None':
            return None, 0


        regex.append('(' + tmp + ')')

    return ''.join(regex).replace('()',''), split_size


def generate_regex_with_split(sigma_lst, sub_id, sub_pos_set, sub_neg_set, split_model, count_limit, alphabet_size, return_dict):
    if sigma_lst is not None and any(list(map(lambda x: x[sub_id], sigma_lst))):
        tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
    else:
        tmp, _ = submodels.SoftConciseNormalForm.synthesizer_snort.synthesis(
            Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model,
            prefix_for_neg_test=None, suffix_for_neg_test=None, alphabet_size=alphabet_size)
        tmp = repr(tmp)

    return_dict[sub_id] = tmp

def generate_split_regex_in_parallel(splited_pos, splited_neg, split_model=False, count_limit=1000, alphabet_size=5,
                         data_type='random', sigma_lst=None, submodel='alpharegex'):
    regex = []

    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    splited_pos = list(filter(lambda x: any(x), splited_pos))
    splited_neg = list(filter(lambda x: any(x), splited_neg))

    pos_split_set = []


    for sub_id in range(split_size):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
        for set_idx in range(len(splited_neg)):
            neg.append(splited_neg[set_idx][0])
        if not neg:
            neg.append('')

        if submodel == 'blue_fringe':
            pos = list(map(lambda x:x.replace('!','z'),pos))
            neg = list(map(lambda x: x.replace('!', 'z'), neg))


        sub_pos_set = set(pos)
        sub_neg_set = set(neg)

        pos_split_set.append([sub_pos_set, sub_neg_set])

    
    procs = []
    manager = Manager()
    return_dict = manager.dict()

    for sub_id in range(split_size - 1): 
        proc = Process(target=generate_regex_with_split, args=(sigma_lst, sub_id, pos_split_set[sub_id][0], pos_split_set[sub_id][1], split_model, count_limit, alphabet_size, return_dict))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    prefix = '(' + ')('.join([return_dict[i] for i in range(split_size - 1)]) + ')'
        
    #if submodel == 'blue_fringe':
    #    count_limit = 1000000000
    #    tmp = rpni_synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model, prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
    #    print(tmp)
    #    tmp = str(tmp)
    if submodel == 'alpharegex':
        if data_type == 'random':            
            tmp = repr(submodels.SoftConciseNormalForm.synthesizer.synthesis(Examples(pos=pos_split_set[-1][0], neg=pos_split_set[-1][1]), count_limit, start_with_no_concat=split_model, 
                prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size))
        else:
            tmp, _ = submodels.SoftConciseNormalForm.synthesizer_snort.synthesis(
                Examples(pos=pos_split_set[-1][0], neg=pos_split_set[-1][1]), count_limit, start_with_no_concat=split_model,
                prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
            tmp = repr(tmp)
    elif submodel == 'set2regex':
        pass
    elif submodel == 'regex_generator':
        #print('ss')
        #print(list(sub_neg_set))
        tmp = execute([Ex(list(pos_split_set[-1][0]), list(pos_split_set[-1][1]))])
        #print(tmp)
        tmp = str(tmp).replace('++', '+').replace('?+', '+')
        #print(tmp)

    if tmp == 'None':
        return None, 0


    final = prefix + '(' + tmp + ')'

    return final, split_size
