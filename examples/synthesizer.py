import os
import argparse
import logging

import time
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchtext

import seq2seq
from seq2seq.trainer.supervised_trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.string_preprocess import get_set_num, get_regex_list
import seq2seq.dataset.dataset

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Synthesizer usage:
#     # synthesize regex from positive, negative strings
#     python examples/synthesizer.py --data_path $DATA_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to data')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    train_path = "../data/train.csv"
    valid_path = "../data/valid.csv"

    batch_size = 256

    data_loader = seq2seq.dataset.dataset.get_loader(train_path, batch_size=batch_size, shuffle=False)

    # Partitioning positive strings
    latest_check_point = Checkpoint.get_latest_checkpoint(opt.checkpoint)
    checkpoint = Checkpoint.load(latest_check_point)
    model = checkpoint.model


    softmax_list, _, other = model(input_variables, input_lengths)

    splitted_pos = split_strings(data_loader)

    # generate subregex


    # Prepare loss
    loss = NLLLoss()
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 512
        bidirectional = opt.bidirectional
        encoder = EncoderRNN(12, 10, hidden_size, dropout_p=0.25, input_dropout_p=0.25,
                             bidirectional=bidirectional, n_layers=2, variable_lengths=True)
        decoder = DecoderRNN(12, 10, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0.2, input_dropout_p=0.25, use_attention=True, bidirectional=bidirectional, n_layers=2,
                             attn_mode=opt.attn_mode)

        seq2seq = Seq2seq(encoder, decoder)
        
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.1, 0.1)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr=0.001), max_grad_norm=0.5)
        scheduler = ReduceLROnPlateau(optimizer.optimizer, 'min', factor=0.1, verbose=True, patience=10)
        optimizer.set_scheduler(scheduler)
    expt_dir = opt.expt_dir + '_hidden_{}'.format(hidden_size)


    # train
    t = SupervisedTrainer(loss=loss, batch_size=batch_size,
                          checkpoint_every=1800,
                          print_every=100, expt_dir=expt_dir)
    
    start_time = time.time()
    seq2seq = t.train(seq2seq, train,
                      num_epochs=50, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)
    end_time = time.time()
    print('total time > ', end_time-start_time)
    
predictor = Predictor(seq2seq, input_vocab, output_vocab)