from __future__ import print_function, division

from collections import Counter
import re
import torch

from seq2seq.loss import NLLLoss
from seq2seq.dataset.dataset import decomposing_regex


def list_chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


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
        total = 0
        correct_seq_re = 0
        correct_set_re = 0

        with torch.no_grad():
            for inputs, outputs, regex in data:

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

                # acc of comparing to regex
                for batch_idx in range(len(regex)):
                    set_count = 0
                    for set_idx in range(10):
                        start = 0
                        all_match = True

                        src_seq = inputs[batch_idx, set_idx].tolist()  # list of 10 alphabet
                        predict_seq_dict = predict_dict[batch_idx * 10 + set_idx]  # predict label. ex. {0.0: 2, 1.0: 1, 11.0: 7}

                        for regex_idx, subregex in enumerate(regex[batch_idx]):
                            if float(regex_idx) in predict_seq_dict:
                                len_subregex = predict_seq_dict[float(regex_idx)]
                                predict_subregex = ''.join([str(a) for a in src_seq[start:start + len_subregex]])
                                start = len_subregex
                            else:
                                predict_subregex = ''

                            # print(subregex, predict_subregex, re.fullmatch(subregex, predict_subregex))
                            if re.fullmatch(subregex, predict_subregex) is None:
                                all_match = False
                        if all_match:
                            correct_seq_re += 1
                            set_count += 1
                    if set_count == 10:
                        correct_set_re += 1
                        # print()

                # acc of comparing to input strings & loss calculating
                for step, step_output in enumerate(decoder_outputs):
                    target = tgt_variables[:, step].to(device='cuda')  # 총 10개의 스텝
                    loss.eval_batch(step_output.view(tgt_variables.size(0), -1), target)

                    if step == 0:
                        match_seq = seqlist[step].view(-1).eq(target).unsqueeze(-1)
                    else:
                        match_seq = torch.cat((match_seq, seqlist[step].view(-1).eq(target).unsqueeze(-1)), dim=1)

                    non_padding = target.ne(11)
                    match += seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    total += non_padding.sum().item()

                result = torch.logical_or(match_seq, tgt_variables.eq(11).to(device='cuda'))
                match_seqnum += [example.all() for example in result].count(True)

                tmp = list_chunk([example.all() for example in result], 10)
                match_setnum += [all(example) for example in tmp].count(True)

        acc_seq = match_seqnum / 200000
        acc_seq_re = correct_seq_re / 200000
        acc_set = match_setnum / 20000
        acc_set_re = correct_set_re / 20000

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy, acc_seq, acc_seq_re, acc_set, acc_set_re
