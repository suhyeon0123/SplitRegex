#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

from collections import Counter
import torch
import torchtext
import argparse
import time
import seq2seq
import subprocess
import re
from seq2seq.dataset.dataset import decomposing_regex, Vocabulary


from regexDFAEquals import regex_equiv_from_raw, unprocess_regex, regex_equiv
from seq2seq.optim import Optimizer
from seq2seq.models import EncoderRNN, DecoderRNN,Seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.evaluator import Predictor
from seq2seq.util.string_preprocess import get_set_num,pad_tensor, count_star, decode_tensor_input, decode_tensor_target
from seq2seq.util.regex_operation import pos_membership_test, neg_membership_test, preprocess_regex, regex_equal, regex_inclusion, valid_regex, or_exception
import seq2seq.dataset.dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path', help='path to data')
parser.add_argument('--checkpoint', action='store', dest='checkpoint', help='path to checkpoint')
opt = parser.parse_args()

opt.checkpoint = '../_hidden_512/best_model/checkpoints'
latest_check_point = Checkpoint.get_latest_checkpoint(opt.checkpoint)
checkpoint = Checkpoint.load(latest_check_point)


model = checkpoint.model
optimizer = checkpoint.optimizer
loss = NLLLoss()
batch_size = 1 
print(model)


data = seq2seq.dataset.dataset.get_loader(opt.data_path, batch_size=batch_size, shuffle=False)
device = torch.device('cuda:0') if torch.cuda.is_available() else -1


model.eval()
loss.reset()
start = time.time()

match = 0
total = 0
num_samples = 0


with torch.no_grad():
    with open('{}_error_analysis.txt'.format(opt.checkpoint), 'w') as fw:
        statistics = [{'cnt':0,'hit': 0,'string_equal':0,'dfa_equal':0, 'membership_equal':0, 'invalid_regex':0} for _ in range(4)]

        for inputs, outputs, regex in data:
            num_samples = num_samples + 1

            # data preprocessing
            for batch_idx in range(len(inputs)):
                inputs[batch_idx] = torch.stack(inputs[batch_idx], dim=0)
                outputs[batch_idx] = torch.stack(outputs[batch_idx], dim=0)

            inputs = torch.stack(inputs, dim=0)
            outputs = torch.stack(outputs, dim=0)

            inputs = inputs.permute(2, 0, 1)
            outputs = outputs.permute(2, 0, 1)

            decoder_outputs, decoder_hidden, other = model(inputs, None, outputs)
            tgt_variables = outputs.contiguous().view(-1, 10)
            tgt_variables = tgt_variables.view(-1, 10)

            regex = list(map(lambda x: decomposing_regex(x), regex))

            answer_dict = [dict(Counter(l)) for l in tgt_variables.tolist()]

            seqlist = other['sequence']
            seqlist2 = [i.tolist() for i in seqlist]
            tmp = torch.Tensor(seqlist2).transpose(0, 1).squeeze(-1).tolist()
            predict_dict = [dict(Counter(l)) for l in tmp]



            # generate output
            with torch.no_grad():
                softmax_list, _, other =model(inputs, tgt_variables)


            vocab = Vocabulary()

            length = other['length'][0]
            tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
            tgt_seq = [vocab.itos[tok] for tok in tgt_id_seq]

            # calculate NLL accuracy
            non_padding = tgt_variables.view(-1)[1:].ne(11).type(torch.LongTensor)
            predict_var = torch.stack(tgt_id_seq)
            target_var = tgt_variables.view(-1)[1:]
            max_len = max(len(predict_var), len(target_var))
            padded_predict_var = pad_tensor(predict_var, max_len, vocab)
            padded_target_var = pad_tensor(target_var, max_len, vocab)
            padded_non_padding = pad_tensor(non_padding, max_len, vocab).type(torch.bool)
            correct = padded_predict_var.eq(padded_target_var).masked_select(padded_non_padding).sum().item()

            match += correct
            total += non_padding.sum().item()

            predict_regex = ' '.join(tgt_seq[:-1])
            target_regex = decode_tensor_target(tgt_variables, vocab)
            target_tokens = target_regex.split()
            predict_tokens = predict_regex.split()
            target_regex, predict_regex = preprocess_regex(target_regex, predict_regex)
            
            pos_input =  decode_tensor_input(inputs, vocab)

            star_cnt = count_star(target_regex)
            statistics[star_cnt]['cnt'] +=1
            
            if not valid_regex(predict_regex) or not or_exception(predict_regex):
                statistics[star_cnt]['invalid_regex'] +=1
            else:    
                if target_regex == predict_regex:
                    statistics[star_cnt]['hit'] +=1
                    statistics[star_cnt]['string_equal']+=1
                elif regex_equal(target_regex, predict_regex):
                    statistics[star_cnt]['hit'] +=1
                    statistics[star_cnt]['dfa_equal'] +=1
                elif pos_membership_test(predict_regex, pos_input):
                    statistics[star_cnt]['hit'] +=1 
                    statistics[star_cnt]['membership_equal'] +=1
                else: 
                    fw.write('pos_input : ' + ' '.join(pos_input)+'\n')
                    fw.write('target_regex : ' + target_regex  +'\n')
                    fw.write('predict_regex : ' + predict_regex + '\n\n')
            
            if total == 0:
                accuracy = float('nan')
            else:
                accuracy = match / total
                
            if num_samples % 100 == 0:
                print("Iterations: ", num_samples, ", " , "{0:.3f}".format(accuracy) + '\n')
            
end = time.time()
print(statistics)
print(end-start)

