import os
import argparse
import logging

import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from seq2seq.trainer.supervised_trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, EncoderRNN2
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.seed import seed_all
import configparser
import seq2seq.dataset.dataset as dataset


# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='./data/random_train.csv', dest='train_path',
                    help='Path to train data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./saved_models',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--bidirectional', action='store_true', dest='bidirectional',
                    default=True,
                    help='Indicates if training model is bidirectional model or not')
parser.add_argument('--num_gpu', action='store', dest='num_gpu',
                    default='0',
                    help='Indicates gpu number')


parser.add_argument('--use_attn', action='store_true', dest='use_attn', default=True, help='use attention or not')
parser.add_argument('--attn_mode', action='store_true', dest='attn_mode', default=False, help='choose attention mode')


opt = parser.parse_args()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

s2smodel = None
input_vocab = None
output_vocab = None

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    s2smodel = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    seed_all(int(config['seed']['train']))

    device = torch.device(f'cuda:{int(opt.num_gpu)}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Prepare dataset
    train_path = opt.train_path

    batch_size = 16384

    if 'random' in opt.train_path:
        MAX_SEQUENCE_LENGTH = 10
    else:
        MAX_SEQUENCE_LENGTH = 15

    train = dataset.get_loader(train_path, batch_size=batch_size, object='train', shuffle=True, max_len=MAX_SEQUENCE_LENGTH)
    dev = dataset.get_loader(train_path, batch_size=batch_size, object='valid', shuffle=False, max_len=MAX_SEQUENCE_LENGTH)

    input_vocab = train.dataset.vocab
    output_vocab = train.dataset.vocab


    rnn_cell = 'gru'


    # Prepare loss
    loss = NLLLoss()
    if torch.cuda.is_available():
        loss.cuda()

    s2smodel = None
    optimizer = None
    if not opt.resume:
        # Initialize model

        hidden_size = 64
        n_layers = 1

        bidirectional = opt.bidirectional
        encoder = EncoderRNN(
                len(input_vocab), dataset.NUM_EXAMPLES, hidden_size,
                dropout_p=0.25, input_dropout_p=0.25,
                bidirectional=bidirectional, n_layers=n_layers, rnn_cell=rnn_cell,
                variable_lengths=True)
        decoder = DecoderRNN(
                len(input_vocab), dataset.NUM_EXAMPLES, hidden_size * (2 if bidirectional else 1),
                dropout_p=0.2, input_dropout_p=0.25, use_attention=True,
                bidirectional=bidirectional, rnn_cell=rnn_cell, n_layers=n_layers, attn_mode=opt.attn_mode)

        s2smodel = Seq2seq(encoder, decoder)

        if torch.cuda.is_available():
            s2smodel.cuda()

        for param in s2smodel.parameters():
            param.data.uniform_(-0.1, 0.1)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.

        optimizer = Optimizer(torch.optim.Adam(s2smodel.parameters(), lr=0.001), max_grad_norm=0.5)
        scheduler = ReduceLROnPlateau(optimizer.optimizer, 'min', factor=0.1, verbose=True, patience=10)
        optimizer.set_scheduler(scheduler)
    expt_dir = opt.expt_dir + '/rnntype_{}_hidden_{}_layer_{}'.format(rnn_cell, hidden_size, n_layers)


    # train
    t = SupervisedTrainer(loss=loss, batch_size=batch_size,
                          checkpoint_every=1800,
                          print_every=100, expt_dir=expt_dir, max_sequence_length=MAX_SEQUENCE_LENGTH)

    start_time = time.time()
    s2smodel = t.train(s2smodel, train,
                      num_epochs=50, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)
    end_time = time.time()
    print('total time > ', end_time-start_time)

predictor = Predictor(s2smodel, input_vocab, output_vocab)
