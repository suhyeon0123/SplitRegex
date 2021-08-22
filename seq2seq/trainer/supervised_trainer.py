from __future__ import division
import logging
import os
import random
import re
import numpy as np
import torch
from torch import optim
from collections import Counter

from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.visualize import visualize_loss
from seq2seq.dataset.dataset import batch_preprocess
from seq2seq.trainer.EarlyStopping import EarlyStopping

def list_chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100, input_vocab = None, output_vocab = None):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size, input_vocab=input_vocab)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.match = 0
        self.match_seqnum = 0
        self.match_setnum = 0
        self.total = 0
        self.correct_seq_re = 0
        self.correct_set_re = 0

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, regex, model, teacher_forcing_ratio):
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, input_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)
        target_variable = target_variable.contiguous().view(-1, 10)

        # Get loss
        loss.reset()

        # acc of comparing to input strings & loss calculating
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

                src_seq = input_variable[batch_idx, set_idx].tolist()  # list of 10 alphabet
                predict_seq_dict = predict_dict[
                    batch_idx * 10 + set_idx]  # predict label. ex. {0.0: 2, 1.0: 1, 11.0: 7}

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
                    self.correct_seq_re += 1
                    set_count += 1
            if set_count == 10:
                self.correct_set_re += 1
                # print()

        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            target = target_variable[:, step].to(device='cuda')  # 총 10개의 스텝
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target)

            if step == 0:
                match_seq = seqlist[step].view(-1).eq(target).unsqueeze(-1)
            else:
                match_seq = torch.cat((match_seq, seqlist[step].view(-1).eq(target).unsqueeze(-1)), dim=1)

            non_padding = target.ne(11)
            self.match += seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
            self.total += non_padding.sum().item()

        result = torch.logical_or(match_seq, target_variable.eq(11).to(device='cuda'))
        self.match_seqnum += [example.all() for example in result].count(True)

        tmp = list_chunk([example.all() for example in result], 10)
        self.match_setnum += [all(example) for example in tmp].count(True)

        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        #device = torch.device('cuda:0') if torch.cuda.is_available() else -1

        steps_per_epoch = len(data)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        best_acc = 0

        # to track the training loss as the model trains
        train_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validtation loss per epoch as the model trains
        avg_valid_losses = []
        early_stopping = EarlyStopping(patience=10, verbose=True)

        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            self.match = 0
            self.match_seqnum = 0
            self.match_setnum = 0
            self.total = 0
            self.correct_seq_re = 0
            self.correct_set_re = 0
            self.total_data_size = 0

            model.train(True)
            for inputs, outputs, regex in data:

                step += 1
                step_elapsed += 1
                self.total_data_size += len(regex)

                inputs, outputs, regex = batch_preprocess(inputs, outputs, regex)

                loss = self._train_batch(inputs.to(device="cuda"), None, outputs, regex, model, teacher_forcing_ratio)

                train_losses.append(loss)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)

            # clear lists to track next epoch
            train_losses = []
            if step_elapsed == 0 : continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0

            print(self.total_data_size)


            accuracyT = self.match / self.total
            acc_seqT = self.match_seqnum / (self.total_data_size * 10)
            acc_seq_reT = self.correct_seq_re / (self.total_data_size * 10)
            acc_setT = self.match_setnum / self.total_data_size
            acc_set_reT = self.correct_set_re / self.total_data_size
            train_log = "Train %s: %.4f, Accuracy: %.4f, Accuracy of seq: %.4f, Accuracy of seq(RE): %.4f, Accuracy of set: %.4f, Accuracy of set(RE): %.4f" % (self.loss.name, epoch_loss_avg, accuracyT, acc_seqT, acc_seq_reT, acc_setT, acc_set_reT)

            if dev_data is not None:
                dev_loss, accuracy, acc_seq, acc_seq_re, acc_set, acc_set_re = self.evaluator.evaluate(model, dev_data)
                avg_valid_losses.append(dev_loss)
                valid_log = "Dev %s: %.4f, Accuracy: %.4f, Accuracy of seq: %.4f, Accuracy of seq(RE): %.4f, Accuracy of set: %.4f, Accuracy of set(RE): %.4f" % (self.loss.name, dev_loss, accuracy, acc_seq, acc_seq_re, acc_set, acc_set_re)
                early_stopping(dev_loss, model, self.optimizer, epoch, step, self.input_vocab, self.output_vocab, self.expt_dir)
                self.optimizer.update(dev_loss, epoch)
                if accuracy > best_acc:
                    log.info('accuracy increased >> best_accuracy{}, current_accuracy{}'.format(accuracy, best_acc))
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=self.input_vocab,
                               output_vocab=self.output_vocab).save(self.expt_dir +'/best_accuracy')
                    best_acc = accuracy
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            log.info('Finished epoch %d:' % epoch)
            log.info(train_log)
            log.info(valid_log)
        return avg_train_losses, avg_valid_losses


    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))
        train_loss, valid_loss = self._train_epoches(data, model, num_epochs,
                                                     start_epoch, step, dev_data=dev_data,
                                                     teacher_forcing_ratio=teacher_forcing_ratio)
        visualize_loss(train_loss, valid_loss, self.expt_dir)
        return model
