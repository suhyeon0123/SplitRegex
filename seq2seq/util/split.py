import torch

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'submodels', 'SoftConsiceNormalFrom' )))

from collections import Counter
from submodels.SoftConsiceNormalFrom.synthesizer import synthesis
from submodels.SoftConsiceNormalFrom.examples import Examples

def split(strings, label, no_split=False):
    
    batch = []
    
    if no_split:
        seq = []
        for batch_idx in range(len(strings)):
            set = []                
            for set_idx in range(10):
                seq.append(''.join(map(str, strings[batch_idx, set_idx][strings[batch_idx, set_idx] != strings.max()].tolist())))
            
            set.append(seq)
        batch.append(set)
        return batch
    
    
    label = [i.tolist() for i in label]
    tmp = torch.LongTensor(label).transpose(0, 1).squeeze(-1).tolist()
    predict_dict = [dict(Counter(l)) for l in tmp]

    for batch_idx in range(len(strings)):
        set = []                
        for set_idx in range(10):
            
            split_size = torch.tensor(label)[torch.tensor(label) != 10].max().item() + 1
            
            src_seq = strings[batch_idx, set_idx].tolist()  # list of 10 alphabet
            #print(src_seq)
            predict_seq_dict = predict_dict[batch_idx * 10 + set_idx]  # predict label. ex. {0.0: 2, 1.0: 1, 11.0: 7}            
            seq = []
            idx = 0
            for seq_id in range(split_size):
                tmp = ''
                if seq_id in predict_seq_dict.keys():
                    for _ in range(predict_seq_dict[float(seq_id)]):
                        tmp += str(src_seq[idx])
                        idx += 1
                seq.append(tmp)
            #print(seq)
            set.append(seq)
        batch.append(set)

    return batch

def generate_split_regex(splited_pos, splited_neg):        
    regex = []
    for sub_id in range(len(splited_pos[0])):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
            if len(splited_neg[set_idx]) > sub_id:
                neg.append(splited_neg[set_idx][sub_id])
            else:
                neg.append('')

        sub_pos_set = set(list(filter(lambda x : x != '', pos)))
        sub_neg_set = set(list(filter(lambda x : x != '', neg)))

        sub_neg_set -= sub_pos_set
        
        print('Splited Positive Strings:', sub_pos_set)
        print('Splited Negative Strings:', sub_neg_set)

        tmp = synthesis(Examples(pos=sub_pos_set, neg=sub_neg_set), 5000)
        if tmp is None:
            return None
        regex.append(tmp)

    return ''.join(regex)







