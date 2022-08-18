import torch

import os, sys
from multiprocessing import Process, Manager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'SoftConciseNormalForm')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'RegexGenerator')))


from collections import Counter
from submodels.RegexGenerator.batch import *
import submodels.SCNF.synthesizer
from submodels.SCNF.parsetree import *

import submodels.SCNF.synthesizer_snort
from submodels.SCNF.util_snort import *

from seq2seq.dataset.dataset import Vocabulary
from submodels.SCNF.examples import Examples
from rpni import synthesis as rpni_synthesis
from synthesis import TimeOutException


class Ex():
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg
    def __str__(self):
        print(self.pos, self.neg)

def is_last_sigma(lst, split_size):

    try:
        idx = len(lst) - 1 - list(reversed(lst)).index(split_size)
    except:
        return False

    if idx != 9 and lst[idx+1] == 0:
        return True

org2RG = {'A':'\.', 'B':':','C':',', 'D':';', 'E':'_', 'F':'=', 'G':'[', 'H':']', 'I':'/', 'J':'\?','K':'\!', 'L':'\{','M':'\}','N':'\(','O':'\)','P':'\<'}
RG2org = {v: k for k, v in org2RG.items()}

# change original string to RG formed string
def get_org2RG(string):
    for k, v in org2RG.items():
        string = string.replace(k, v)
    return string
    
def get_RG2org(string):
    for k, v in RG2org.items():
        string = string.replace(k, v)
    return string

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


# for subregex synthesis with baselines
def generate_regex_with_split_ar(sigma_lst, sub_id, sub_pos_set, sub_neg_set, split_model, count_limit, prefix, alphabet_size, data_type, return_dict):
    if len(sub_pos_set) == 1:
        return_dict[sub_id] = sub_pos_set.pop()
        return
        
    if data_type == 'random':
        if sigma_lst is not None  and any(list(map(lambda x: x[sub_id], sigma_lst))):
            tmp = repr(KleenStar(Or(*[Character(str(x)) for x in range(alphabet_size)])))
        else:
            tmp = repr(submodels.SCNF.synthesizer.synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set),
                                                                        count_limit,
                                                                        start_with_no_concat=split_model,
                                                                        prefix_for_neg_test=prefix,
                                                                        suffix_for_neg_test=None,
                                                                            alphabet_size=alphabet_size))
    else:
        if sigma_lst is not None  and any(list(map(lambda x: x[sub_id], sigma_lst))):
            tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
        else:
            tmp, _ = submodels.SCNF.synthesizer_snort.synthesis(
                Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model,
                prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)
            tmp = repr(tmp)

    return_dict[sub_id] = tmp

def generate_regex_with_split_bf(sub_id, sub_pos_set, sub_neg_set, split_model, count_limit, prefix, alphabet_size, return_dict):

    if len(sub_pos_set) == 1:
        return_dict[sub_id] = sub_pos_set.pop()
        return

    tmp = rpni_synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), count_limit, start_with_no_concat=split_model, prefix_for_neg_test=prefix, suffix_for_neg_test=None, alphabet_size=alphabet_size)

    return_dict[sub_id] = str(tmp)

def generate_regex_with_split_rg(sigma_lst, sub_id, sub_pos_set, sub_neg_set, return_dict):   
    
    if len(sub_pos_set) == 1:
        return_dict[sub_id] = sub_pos_set.pop()
        return

    # print(sub_pos_set, sub_neg_set)
    # new_pos_set = set()
    # for x in sub_pos_set:
    #     new_pos_set.add(get_org2RG(x))
    # new_neg_set = set()
    # for x in sub_neg_set:
    #     new_neg_set.add(get_org2RG(x))
    # print(new_pos_set, new_neg_set)

    if sigma_lst is not None and any(list(map(lambda x: x[sub_id], sigma_lst))):
        tmp = get_sigma(Examples(pos=sub_pos_set, neg=sub_neg_set))
    else:
        tmp = execute([Ex(list(sub_pos_set), list(sub_neg_set))])
 
    tmp = str(tmp).replace('++', '+').replace('?+', '+')
    
    # tmp = get_RG2org(tmp)

    return_dict[sub_id] = tmp




def generate_split_regex_sequential(splited_pos, splited_neg, split_model=False, count_limit=1000, alphabet_size=5,
                         data_type='random', sigma_lst=None, submodel='alpharegex', return_dict=None, use_prefix_every=False):

    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    splited_pos = list(filter(lambda x: any(x), splited_pos))
    splited_neg = list(filter(lambda x: any(x), splited_neg))

    split_set = []

    for sub_id in range(split_size):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
        for set_idx in range(len(splited_neg)):
            neg.append(splited_neg[set_idx][0])
        if not neg:
            neg.append('')

        split_set.append([set(pos), set(neg)])

    # synthesis one by one
    for sub_id in range(split_size): 

        # prefix strategy (only nth element or every element)
        if sub_id != 0 and (sub_id == split_size - 1 or use_prefix_every):
            prefix = '(' + ')('.join([return_dict[i] for i in range(sub_id)]) + ')'
        else:
            split_set[sub_id][1] -= split_set[sub_id][0]
            prefix = None

        
        print('Splited Positive Strings:', split_set[sub_id][0])
        print('Splited Negative Strings:', split_set[sub_id][1])

        if submodel == 'alpharegex':
            generate_regex_with_split_ar(sigma_lst, sub_id, split_set[sub_id][0], split_set[sub_id][1], split_model, count_limit, prefix, alphabet_size, data_type, return_dict)
        elif submodel == 'blue_fringe':
            count_limit = 1000000000
            generate_regex_with_split_bf(sub_id, split_set[sub_id][0], split_set[sub_id][1], split_model, count_limit, prefix, alphabet_size, return_dict)
        elif submodel == 'regex_generator':
            generate_regex_with_split_rg(sigma_lst, sub_id, split_set[sub_id][0], split_set[sub_id][1], return_dict)
        else:
            raise Exception('unknown baseline')

    
    return '(' + ')('.join([return_dict[i] for i in range(split_size)]) + ')', split_size
    


def generate_split_regex_in_parallel(splited_pos, splited_neg, split_model=False, count_limit=1000, alphabet_size=5,
                         data_type='random', sigma_lst=None, submodel='alpharegex', return_dict=None, use_prefix_every=False):

    split_size = len(splited_pos[0])
    print("Split Size: ", split_size)

    splited_pos = list(filter(lambda x: any(x), splited_pos))
    splited_neg = list(filter(lambda x: any(x), splited_neg))

    split_set = []
    procs = []

    for sub_id in range(split_size):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
        for set_idx in range(len(splited_neg)):
            neg.append(splited_neg[set_idx][0])
        if not neg:
            neg.append('')

        split_set.append([set(pos), set(neg)])

    
    # parallel for regex_generator
    try:
        if submodel == 'regex_generator':
            for sub_id in range(split_size): 
                proc = Process(target=generate_regex_with_split_rg, args=(sigma_lst, sub_id, split_set[sub_id][0], split_set[sub_id][1], return_dict))

                procs.append(proc)
                proc.start()

            for proc in procs:
                proc.join()

            return '(' + ')('.join([return_dict[i] for i in range(split_size)]) + ')'
    except Exception as e:
        for proc in procs:
            proc.terminate()
        raise TimeOutException()


    # parallel synthesis [1, n-1]
    try:
        prefix = None
        for sub_id in range(split_size - 1): 
            if submodel == 'alpharegex':
                proc = Process(target=generate_regex_with_split_ar, args=(sigma_lst, sub_id, split_set[sub_id][0], split_set[sub_id][1], split_model, count_limit, prefix, alphabet_size, data_type, return_dict))
            elif submodel == 'blue_fringe':
                count_limit = 1000000000
                proc = Process(target=generate_regex_with_split_bf, args=(sub_id, split_set[sub_id][0], split_set[sub_id][1], split_model, count_limit, prefix, alphabet_size, return_dict))
            else:
                raise Exception('unknown baseline')

            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()
    except Exception as e:
        for proc in procs:
            proc.terminate()
        print('catch processes')
        raise TimeOutException()

    # synthesis for nth subregex
    if split_size > 1:
        prefix = '(' + ')('.join([return_dict[i] for i in range(split_size - 1)]) + ')'
    else:
        prefix = None
        
    if submodel == 'alpharegex':
        generate_regex_with_split_ar(sigma_lst, split_size-1, split_set[split_size-1][0], split_set[split_size-1][1], split_model, count_limit, prefix, alphabet_size, data_type, return_dict)
    elif submodel == 'blue_fringe':
        count_limit = 1000000000
        generate_regex_with_split_bf(split_size-1, split_set[split_size-1][0], split_set[split_size-1][1], split_model, count_limit, prefix, alphabet_size, return_dict)
    else:
        raise Exception('unknown baseline')

    return '(' + ')('.join([return_dict[i] for i in range(split_size)]) + ')', split_size
