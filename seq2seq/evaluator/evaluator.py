from __future__ import print_function, division

from collections import Counter
import re
import numpy as np
import torch
import torchtext
from seq2seq.dataset import SourceField

import seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.util.string_preprocess import pad_tensor
from seq2seq.util.utils import decomposing_regex

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]





class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64, input_vocab=None):
        self.loss = loss
        self.batch_size = batch_size
        self.input_vocab = input_vocab

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        match_seqnum = 0
        match_setnum = 0
        match_seqnum2 = 0
        match_setnum2 = 0
        only_pad_count = 0
        only_pad_count2 = 0
        total = 0
        correct_seq_re = 0

        with torch.no_grad():
            for inputs, outputs, regex in data:

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


                regex = list(map(lambda x:decomposing_regex(x), regex))
                print()
                print(regex[:10])

                answer_dict = [dict(Counter(l)) for l in tgt_variables.tolist()]
                print(answer_dict[:10])

                '''
                example = []
                src_variables1 = inputs.contiguous().view(-1, 10)
                batch_size = tgt_variables.size(0)
                for batch_idx in range(batch_size):
                    batch_example = {}
                    example_idx = 0
                    for label in range(2, 12):
                        if label in result[batch_idx]:
                            batch_example[label - 2] = src_variables1[batch_idx][
                                                       example_idx:example_idx + result[batch_idx][label]]
                            example_idx += result[batch_idx][label]

                    example.append(batch_example)'''

                seqlist = other['sequence']
                seqlist2 = [i.tolist() for i in seqlist]
                tmp = torch.Tensor(seqlist2).transpose(0, 1).squeeze(-1).tolist()
                predict_dict = [dict(Counter(l)) for l in tmp]
                print(predict_dict[:10])

                '''example = []
                for batch_idx in range(batch_size):
                    batch_example = {}
                    example_idx = 0
                    for label in range(2, 12):
                        if label in result[batch_idx]:
                            batch_example[label - 2] = src_variables1[batch_idx][
                                                       example_idx:example_idx + result[batch_idx][label]]
                            example_idx += result[batch_idx][label]
                    example.append(batch_example)
'''

                #check the re fullmatching
                '''src = inputs.view(-1, 10).tolist()
                for idx, count_dict in enumerate(result):
                    target_full_regex = idx_variables[idx % 10]
                    print('target_full_regex:', target_full_regex)
                    embed = 2.0
                    src_seq = list(map(lambda x: x-2, src[idx]))
                    end = 0
                    total = 0
                    limit = src_seq.count(1) + src_seq.count(0)
                    for sub_regex in target_full_regex:
                        try:
                            key = count_dict[embed]
                        except KeyError:
                            break
                        total += key
                        if total > limit:
                            key -= total - limit
                        start = end
                        end = start + key
                        embed += 1
                        print('sub_regex', sub_regex)
                        print('src_seq', src_seq[start:end])
                        print(re.fullmatch(sub_regex, ''.join([str(a) for a in src_seq[start:end]])))
                '''

                # Evaluation
                '''
                match_seq = tgt_variables.eq(torch.Tensor(torch.Tensor(seqlist2).transpose(0, 1).squeeze(-1).tolist()).to("cuda"))

                result = torch.logical_or(match_seq, tgt_variables.eq(pad))
                #print([example.all() for example in result])
                match_seqnum2 += [example.all() for example in result].count(True)
                only_pad_count2 += [example.all() for example in tgt_variables.eq(pad)].count(True)

                tmp = [example.all() for example in result]
                tmp = list_chunk(tmp, 10)
                match_setnum2 += [all(example) for example in tmp].count(True)
                #print([all(example) for example in tmp])
                acc_seq2 = (match_seqnum2 - only_pad_count2) / (20000 * 10 - only_pad_count2)
                acc_set2 = match_setnum2 / 20000'''


                '''match_seq = torch.zeros(len(tgt_variables)) #640
                answer = tgt_variables.to(device='cuda')
                #predict = decoder_outputs.reshape(len(tgt_variables),10,12)
                predict = np.array(decoder_outputs).reshape(len(tgt_variables),10,12).tolist()  #640,10,12
                seqlist2 = [i.tolist() for i in seqlist]
                seqlist2 = torch.Tensor(seqlist2).transpose(0, 1).squeeze(-1).tolist()'''


                print(regex)
                print(answer_dict)
                print(predict_dict)
                print(len(regex))
                print(len(answer_dict))
                print(len(predict_dict))



                for batch_idx in range(len(regex)):
                    batch_regex = regex[batch_idx]
                    for set_idx in range(10):
                        start = 0
                        all_match = True

                        src_seq = inputs[batch_idx, set_idx].tolist()
                        set_dict = predict_dict[batch_idx*10 + set_idx]
                        #print(src_seq)
                        #print(set_dict)
                        for regex_idx, subregex in enumerate(batch_regex):
                            if float(regex_idx) in set_dict:
                                len_subregex = set_dict[float(regex_idx)]
                                predict_subregex = ''.join([str(a) for a in src_seq[start:start + len_subregex]])
                                start = len_subregex
                            else:
                                predict_subregex = ''

                            #print(subregex, predict_subregex, re.fullmatch(subregex, predict_subregex))
                            if re.fullmatch(subregex, predict_subregex) is None:
                                all_match = False
                        if all_match:
                            correct_seq_re += 1
                        #print()


                for step, step_output in enumerate(decoder_outputs):
                    target = tgt_variables[:, step].to(device='cuda')  # 총 10개의 스텝
                    loss.eval_batch(step_output.view(tgt_variables.size(0), -1), target)

                    if step == 0:
                        match_seq = seqlist[step].view(-1).eq(target).unsqueeze(-1)
                    else:
                        match_seq = torch.cat((match_seq, seqlist[step].view(-1).eq(target).unsqueeze(-1)), dim=1)
                    non_padding = target.ne(11)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

                #print(match_seq.shape) # 640, 10

                result = torch.logical_or(match_seq, tgt_variables.eq(11).to(device='cuda'))
                #print([example.all() for example in result])
                match_seqnum += [example.all() for example in result].count(True)
                only_pad_count += [example.all() for example in tgt_variables.eq(11)].count(True)

                tmp = [example.all() for example in result]
                tmp = list_chunk(tmp, 10)
                match_setnum += [all(example) for example in tmp].count(True)
                #print([all(example) for example in tmp])

        print(correct_seq_re/200000)
        acc_seq = (match_seqnum - only_pad_count) / (20000 * 10 - only_pad_count)
        acc_set = match_setnum / 20000
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total



        return loss.get_loss(), accuracy, acc_seq, acc_set

