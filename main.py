import argparse
import torch
from seq2seq.dataset import dataset
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.split import split

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to data')
parser.add_argument('--batch_size', action='store', dest='batch_size',
                    help='batch size', default=64)
parser.add_argument('--checkpoint_pos', action='store', dest='checkpoint_pos',
                    help='path to checkpoint for splitting positive strings ')
parser.add_argument('--checkpoint_neg', action='store', dest='checkpoint_neg',
                    help='path to checkpoint for splitting negative strings ')
parser.add_argument('--sub_model', action='store', dest='sub_model', default='set2regex',
                    help='sub model used in generating sub regex from sub strings')

opt = parser.parse_args()

data = dataset.get_loader(opt.data_path, batch_size=opt.batch_size, shuffle=False)
pos_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_pos))
neg_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_neg))

pos_split_model = pos_checkpoint.model
neg_split_model = neg_checkpoint.model

pos_split_model.eval()
neg_split_model.eval()

with torch.no_grad():
    for pos, neg, regex in data:
        pos, neg, regex = dataset.batch_preprocess(pos, neg, regex)

        _, _, other = pos_split_model(pos, None, regex)
        splited_pos = split(pos, other['sequence'])  # batch, set, seq

        _, _, other = neg_split_model(neg)
        splited_neg = split(neg, other['sequence'])  # batch, set, seq

        sub_regex = sub_model(splited_pos, splited_neg)

        predict = sub_regex.sum()
