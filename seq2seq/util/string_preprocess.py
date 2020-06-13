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
