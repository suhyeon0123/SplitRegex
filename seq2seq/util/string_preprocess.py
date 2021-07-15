#!/usr/bin/env python
# coding: utf-8

import torch

def get_set_num(dataset_file):
    with open(dataset_file, 'r') as rf:
        dataset = rf.read().split('\n')
    return len(dataset[0].split('\t'))-1

def get_regex_list(dataset_file):
    with open(dataset_file, 'r') as rf:
        dataset = rf.read().split('\n')
    return dataset

def pad_tensor(input_var, max_len, vocab):
    pad_idx = vocab.stoi['<pad>']
    input_var = input_var.cuda()
    if len(input_var) > 10:
        input_var = input_var[:10]
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


def preprocessing(input_var, none_idx):
    processed_input = []
    set_size_in_batch = []
    input_var = torch.split(input_var, input_var.size(0), dim=0)
    none_idx_within_batch = [(input_var[0][idx] == none_idx).nonzero() for idx in range(len(input_var[0]))]

    for minibatch, idx in zip(input_var[0], none_idx_within_batch):
        if len(idx) ==0:
            processed_input.append(minibatch)
            set_size_in_batch.append(minibatch.size(0))
            continue
        set_size_in_batch.append(idx[0][0].item())
        processed_input.append(torch.narrow(minibatch, 0,0, idx[0][0].item()))

    return torch.cat(processed_input, dim=0), set_size_in_batch


def get_set_lengths(input_var):
    '''
    1-> pad token 
    '''
    return (~(input_var[:,:, 0] == 1)).sum(dim=-1)


def get_mask(inputs):
    masking = torch.eq(inputs, 1)
    return masking


def pad_attention(attn, max_len):
    diff  = max_len - attn.size(-1)
    zero_padding = torch.zeros(attn.size(0), attn.size(1), diff).cuda()
    attn = torch.cat((attn, zero_padding), dim=-1)
    return attn


def get_mask2(set_lens, max_len):
    masking = [torch.tensor((val * [1] + (max_len-val)* [0])) for val in set_lens]
    masking = torch.stack(masking, dim=0)
    masking= ~masking.type(torch.BoolTensor).cuda()
    return masking