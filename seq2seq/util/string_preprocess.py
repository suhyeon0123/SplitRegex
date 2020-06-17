#!/usr/bin/env python
# coding: utf-8

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


def decode_tensor_input(input_var, vocab):
    '''
    decoder input tensor when evaluation step
    '''
    result = []
    input_var = input_var.squeeze(0)
    
    for i in range(input_var.size(0)):
        res = ""
        for j in range(input_var[i].size(0)):
            res += vocab.itos[input_var[i][j]]
        res = res.replace('<pad>', '')
        result.append(res)
        
    return result


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


def stoi(tensors, vocab):
    result = []
    for val in tensors:
        vocab_idx = vocab.stoi[val]
        result.append(vocab_idx)
    return result