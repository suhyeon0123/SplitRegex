#!/usr/bin/env python
# coding: utf-8

# In[11]:
import torch

def get_set_num(dataset_file):
    with open(dataset_file, 'r') as rf:
        dataset = rf.read().split('\n')
    return len(dataset[0].split('\t'))-1


def pad_tensor(input_var, max_len, vocab):
    pad_idx = vocab.stoi['<pad>']
    input_var = input_var.cuda()
    if len(input_var) < max_len:
        padded = torch.Tensor([pad_idx]*(max_len-len(input_var))).type(torch.LongTensor).cuda()
        input_var = torch.cat((input_var, padded))
    return input_var


def count_star(string):
    cnt = 0
    for char in string:
        if char =='*':
            cnt+=1
    return cnt


def decode_tensor_input(batch, input_vocab, set_num):
    input_strings = []
    for num in range(1,set_num+1):
        src_tensor, src_length = getattr(batch, 'src{}'.format(num))
        src_tensor =src_tensor.view(-1)
        strings = ''
        for i in src_tensor:
            word = input_vocab.itos[i]
            strings += word
        strings = strings.replace('<pad>','')
        input_strings.append(strings)
    return input_strings


def decode_tensor_target(tensor, vocab):
    tensor = tensor.view(-1)
    words = []
    for i in tensor:
        word = vocab.itos[i]
        if word == '<eos>':
            return ' '.join(words) 
        if word != '<sos>' and word != '<pad>' and word != '<eos>':
            words.append(word)
    return ' '.join(words)


def regex_equal(regex1, regex2):
    import FAdo.reex
    import FAdo
    regex1 = regex1.replace(' ','')
    regex1 = regex1.replace('[0-3]','(0|1|2|3)')
    regex2 = regex2.replace(' ','')
    regex2 = regex2.replace('[0-3]','(0|1|2|3)')
    dfa1 = FAdo.reex.str2regexp(regex1).toDFA()
    dfa2 = FAdo.reex.str2regexp(regex2).toDFA()
    return dfa1.equal(dfa2)

