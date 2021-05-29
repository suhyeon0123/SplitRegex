from __future__ import print_function, division

from collections import Counter

import numpy as np
import torch
import torchtext
from seq2seq.dataset import SourceField

import seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.util.string_preprocess import pad_tensor

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
        only_pad_count = 0
        match_setnum = 0
        total = 0

        device = torch.device('cuda:0') if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=False,
            device=device, repeat=False, shuffle=True, train=False)


        tgt_vocab = data.fields['tgt1'].vocab
        pad = tgt_vocab.stoi[data.fields['tgt1'].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:

                src_variables = [[] for i in range(batch.batch_size)]
                tgt_variables = [[] for i in range(batch.batch_size)]
                lengths = [[] for i in range(batch.batch_size)]

                set_size = len(batch.fields) / 2
                max_len_within_batch = -1

                for idx in range(batch.batch_size):
                    for src_idx in range(1, int(set_size) + 1):
                        src, src_len = getattr(batch, 'src{}'.format(src_idx))
                        src_variables[idx].append(src[idx])
                        tgt, tgt_len = getattr(batch, 'tgt{}'.format(src_idx))
                        tgt_variables[idx].append(tgt[idx])
                        lengths[idx].append(src_len[idx])

                    lengths[idx] = torch.stack(lengths[idx], dim=0)

                    if max_len_within_batch < torch.max(lengths[idx].view(-1)).item():
                        max_len_within_batch = torch.max(lengths[idx].view(-1)).item()

                for batch_idx in range(len(src_variables)):
                    for set_idx in range(int(set_size)):
                        src_variables[batch_idx][set_idx] = pad_tensor(src_variables[batch_idx][set_idx],
                                                                       max_len_within_batch, self.input_vocab)

                        tgt_variables[batch_idx][set_idx] = pad_tensor(tgt_variables[batch_idx][set_idx],
                                                                       max_len_within_batch, tgt_vocab)

                    src_variables[batch_idx] = torch.stack(src_variables[batch_idx], dim=0)
                    tgt_variables[batch_idx] = torch.stack(tgt_variables[batch_idx], dim=0)

                # ---- a copy from supervised_trainer.py

                src_variables = torch.stack(src_variables, dim=0)
                tgt_variables = torch.stack(tgt_variables, dim=0)
                lengths = torch.stack(lengths, dim=0)


                decoder_outputs, decoder_hidden, other = model(src_variables, lengths, tgt_variables)
                tgt_variables = tgt_variables.view(-1, 10)




                # debug
                src_variables1 = src_variables.view(-1, 10)
                batch_size = tgt_variables.size(0)

                result = [dict(Counter(l)) for l in tgt_variables.tolist()]
                print("")
                print(result[:10])
                example = []
                for batch_idx in range(batch_size):
                    batch_example = {}
                    example_idx = 0
                    for label in range(2, 12):
                        if label in result[batch_idx]:
                            batch_example[label - 2] = src_variables1[batch_idx][
                                                       example_idx:example_idx + result[batch_idx][label]]
                            example_idx += result[batch_idx][label]

                    example.append(batch_example)

                seqlist = other['sequence']
                seqlist = [i.tolist() for i in seqlist]
                tmp = torch.Tensor(seqlist).transpose(0, 1).squeeze(-1).tolist()
                result = [dict(Counter(l)) for l in tmp]
                print(result[:10])
                example = []
                for batch_idx in range(batch_size):
                    batch_example = {}
                    example_idx = 0
                    for label in range(2, 12):
                        if label in result[batch_idx]:
                            batch_example[label - 2] = src_variables1[batch_idx][
                                                       example_idx:example_idx + result[batch_idx][label]]
                            example_idx += result[batch_idx][label]
                    example.append(batch_example)



                # Evaluation
                seqlist = other['sequence']# 10,640

                match_seq = None
                match_seq = torch.zeros(len(tgt_variables))
                for step, step_output in enumerate(decoder_outputs):
                    target = tgt_variables[:, step]
                    loss.eval_batch(step_output.view(tgt_variables.size(0), -1), target)

                    if step == 0:
                        match_seq = seqlist[step].view(-1).eq(target).unsqueeze(-1)
                    else:
                        match_seq = torch.cat((match_seq, seqlist[step].view(-1).eq(target).unsqueeze(-1)), dim=1)
                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()
                #print(match_seq.shape) # 640, 10

                result = torch.logical_or(match_seq, tgt_variables.eq(pad))
                match_seqnum += [example.all() for example in result].count(True)
                only_pad_count += [example.all() for example in tgt_variables.eq(pad)].count(True)

                tmp = [example.all() for example in result]
                tmp = list_chunk(tmp, 10)
                match_setnum += [all(example) for example in tmp].count(True)



        acc_seq = (match_seqnum - only_pad_count) / (20000 * 10 - only_pad_count)
        acc_set = match_setnum / 20000
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy, acc_seq, acc_set

