import argparse
import torch
from seq2seq.dataset import dataset
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.split import split, generate_split_regex

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/train.csv', dest='data_path',
                    help='Path to data')
parser.add_argument('--batch_size', action='store', dest='batch_size',
                    help='batch size', default=1)
parser.add_argument('--checkpoint_pos', default='./saved_models/hidden_512/best_model', dest='checkpoint_pos',
                    help='path to checkpoint for splitting positive strings ')
parser.add_argument('--checkpoint_neg', default='./saved_models/hidden_512/best_model', dest='checkpoint_neg',
                    help='path to checkpoint for splitting negative strings ')
parser.add_argument('--sub_model', action='store', dest='sub_model', default='set2regex',
                    help='sub model used in generating sub regex from sub strings')

opt = parser.parse_args()





def print_tensor_set(tensor_set):
    output_strings = ''.join(map(str, tensor_set[0][tensor_set[0] != 10].tolist()))
    for i in range(1, tensor_set.shape[0]):
        output_strings += ', ' + ''.join(map(str, tensor_set[i][tensor_set[i] != 10].tolist()))
    
    print(output_strings)

    
def main():    
    data = dataset.get_loader(opt.data_path, batch_size=opt.batch_size, shuffle=False)
    
    pos_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_pos))
    neg_checkpoint = Checkpoint.load(Checkpoint.get_latest_checkpoint(opt.checkpoint_neg))

    pos_split_model = pos_checkpoint.model
    neg_split_model = neg_checkpoint.model

    pos_split_model.eval()
    neg_split_model.eval()

    for pos, neg, regex in data:
        pos, neg, regex = dataset.batch_preprocess(pos, neg, regex)

        print('Positive Strings')
        print_tensor_set(pos[0])
        print('Negative Strings')
        print_tensor_set(neg[0])
        print('Target Regex:', ''.join(regex[0]))

        _, _, other = pos_split_model(pos, None, regex)
        splited_pos = split(pos, other['sequence'])  # batch, set, seq

        _, _, other = neg_split_model(neg)
        splited_neg = split(neg, other['sequence'])  # batch, set, seq

        batch_predict = []
        for batch_idx in range(len(pos)):
            batch_predict.append(generate_split_regex(splited_pos[batch_idx], splited_neg[batch_idx]))

        print('Generated Regex:', batch_predict[0])

        
if __name__ == "__main__":
    main()