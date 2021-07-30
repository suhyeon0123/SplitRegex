import torch
from collections import Counter
from submodels.SoftConsiceNormalFrom.synthesizer import synthesis
from submodels.SoftConsiceNormalFrom.examples import Examples

def split(strings, label):
    label = [i.tolist() for i in label]
    tmp = torch.Tensor(label).transpose(0, 1).squeeze(-1).tolist()
    predict_dict = [dict(Counter(l)) for l in tmp]

    batch = []
    for batch_idx in range(len(strings)):
        set = []
        for set_idx in range(10):
            src_seq = strings[batch_idx, set_idx].tolist()  # list of 10 alphabet
            #print(src_seq)
            predict_seq_dict = predict_dict[batch_idx * 10 + set_idx]  # predict label. ex. {0.0: 2, 1.0: 1, 11.0: 7}
            #print(predict_seq_dict)
            seq = []
            idx = 0
            for seq_id in range(10):
                tmp = ''
                if float(seq_id) in predict_seq_dict.keys():
                    for _ in range(predict_seq_dict[float(seq_id)]):
                        tmp += str(src_seq[idx])
                        idx += 1
                seq.append(tmp)
            #print(seq)
            set.append(seq)
        #print(set)
        batch.append(set)

    return batch

def generate_split_regex(splited_pos, splited_neg):

    print(splited_neg)
    print(splited_pos)

    regex = []
    for sub_id in range(len(splited_pos[0])):
        pos = []
        neg = []

        for set_idx in range(len(splited_pos)):
            pos.append(splited_pos[set_idx][sub_id])
            neg.append(splited_neg[set_idx][sub_id])

        print(list(filter(lambda x : x != '', pos)))
        print(list(filter(lambda x : x != '', neg)))
        tmp = synthesis(Examples(pos=list(filter(lambda x : x != '', pos)), neg=list(filter(lambda x : x != '', pos))), 5000)
        regex.append(tmp)
        print(tmp)

    return ''.join(regex)







