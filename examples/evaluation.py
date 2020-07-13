#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import torch
import torchtext
import argparse
import time
import seq2seq
import subprocess
import re

from regexDFAEquals import regex_equiv_from_raw, unprocess_regex, regex_equiv
from seq2seq.optim import Optimizer
from seq2seq.models import EncoderRNN, DecoderRNN,Seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.string_preprocess import get_set_num,pad_tensor, count_star, decode_tensor_input, decode_tensor_target
from seq2seq.util.regex_operation import membership_test, preprocess_regex, regex_equal, regex_inclusion


# In[ ]:



parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', help='path to train data')
parser.add_argument('--test_path', action='store', dest='test_path', help='path to test data')
parser.add_argument('--checkpoint', action='store', dest='checkpoint', help='path to checkpoint')
opt = parser.parse_args()


# In[76]:


latest_check_point = Checkpoint.get_latest_checkpoint(opt.checkpoint)
checkpoint = Checkpoint.load(latest_check_point)
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

model = checkpoint.model
optimizer = checkpoint.optimizer
weight = torch.ones(len(output_vocab))
pad = output_vocab.stoi['<pad>']
loss = NLLLoss(weight, pad)
batch_size = 1 
print(model)


# In[3]:


train_file = opt.train_path
test_file = opt.test_path
set_num = get_set_num(train_file)
src = SourceField()
tgt = TargetField()

train = torchtext.data.TabularDataset(
    path=train_file, format='tsv',
    fields= [('src{}'.format(i+1), src) for i in range(set_num)]
    +[('tgt', tgt)]
)

test_data = torchtext.data.TabularDataset(
    path=test_file, format='tsv',
    fields= [('src{}'.format(i+1), src) for i in range(set_num)]+[('tgt', tgt)]
)

src.build_vocab(train, max_size=500)
tgt.build_vocab(train, max_size=500)

device = torch.device('cuda:0') if torch.cuda.is_available() else -1
batch_iterator = torchtext.data.BucketIterator(
    dataset=test_data, batch_size=1,
    sort=False, sort_within_batch=False,
    device=device, repeat=False, shuffle=False)


# In[75]:


model.eval()
loss.reset()
start = time.time()

match = 0
total = 0
num_samples = 0

with torch.no_grad():
    with open('{}_error_analysis.txt'.format(opt.checkpoint), 'w') as fw:
        statistics = [{'cnt':0,'hit': 0,'string_equal':0,'dfa_equal':0, 'inclusion_equal':0} for _ in range(4)]
        for batch in batch_iterator:
            num_samples = num_samples + 1
            target_variables = getattr(batch, seq2seq.tgt_field_name)
            input_variables = [[] for i in range(batch.batch_size)]
            input_lengths = [[] for i in range(batch.batch_size)]
            set_size = len(batch.fields)-1
            max_len_within_batch = -1

            for idx in range(batch.batch_size):
                for src_idx in range(1, set_size+1):
                    src, src_len = getattr(batch, 'src{}'.format(src_idx))
                    input_variables[idx].append(src[idx])
                    input_lengths[idx].append(src_len[idx])
                    
                input_lengths[idx] = torch.stack(input_lengths[idx], dim =0)
                if max_len_within_batch <  torch.max(input_lengths[idx].view(-1)).item():
                    max_len_within_batch = torch.max(input_lengths[idx].view(-1)).item()

            for batch_idx in range(len(input_variables)):
                for set_idx in range(set_size):
                    input_variables[batch_idx][set_idx] = pad_tensor(input_variables[batch_idx][set_idx], 
                                                                     max_len_within_batch, input_vocab)
                input_variables[batch_idx] = torch.stack(input_variables[batch_idx], dim=0)
    
            input_variables = torch.stack(input_variables, dim=0)
            input_lengths = torch.stack(input_lengths, dim=0)
            
            with torch.no_grad():
                softmax_list, _, other =model(input_variables, input_lengths)
        
            length = other['length'][0]
            tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
            tgt_seq = [output_vocab.itos[tok] for tok in tgt_id_seq]

            # calculate NLL accuracy
            non_padding = target_variables.view(-1)[1:].ne(pad).type(torch.LongTensor)
            predict_var = torch.stack(tgt_id_seq)
            target_var = target_variables.view(-1)[1:]
            max_len = max(len(predict_var), len(target_var))
            padded_predict_var = pad_tensor(predict_var, max_len, output_vocab)
            padded_target_var = pad_tensor(target_var, max_len, output_vocab)
            padded_non_padding = pad_tensor(non_padding, max_len, output_vocab).type(torch.bool)
            correct = padded_predict_var.eq(padded_target_var).masked_select(padded_non_padding).sum().item()

            match += correct
            total += non_padding.sum().item()

            predict_regex = ' '.join(tgt_seq[:-1])
            target_regex = decode_tensor_target(target_variables, output_vocab)
            target_tokens = target_regex.split()
            predict_tokens = predict_regex.split()
            edited_target_regex, edited_predict_regex = preprocess_regex(target_regex, predict_regex)
            input_words =  decode_tensor_input(input_variables, input_vocab)
            input_words = list(filter(('none').__ne__, input_words))
            star_cnt = count_star(target_regex)
            statistics[star_cnt]['cnt'] +=1
            
            # calculate regex equivalent accuracy
            try:            
                if edited_target_regex == edited_predict_regex:
                    statistics[star_cnt]['hit'] +=1
                    statistics[star_cnt]['string_equal']+=1
                elif regex_equal(edited_target_regex, edited_predict_regex):
                    statistics[star_cnt]['hit'] +=1
                    statistics[star_cnt]['dfa_equal'] +=1
                elif membership_test(edited_predict_regex, input_words) and regex_inclusion(edited_target_regex, edited_predict_regex): # inclusion equal
                    statistics[star_cnt]['hit'] +=1 
                    statistics[star_cnt]['inclusion_equal'] +=1
                else: 
                    fw.write('input_string : ' + ' '.join(input_words)+'\n') 
                    fw.write('target_regex : ' + target_regex  +'\n')
                    fw.write('predict_regex : ' + predict_regex + '\n\n')
            except:
                fw.write('invalid predicted regex >  {}\n'.format(edited_predict_regex))
            
            if total == 0:
                accuracy = float('nan')
            else:
                accuracy = match / total
                
            if num_samples % 100 == 0:
                print("Iterations: ", num_samples, ", " , "{0:.3f}".format(accuracy) + '\n')
            
end = time.time()
print(statistics)
print(end-start)

