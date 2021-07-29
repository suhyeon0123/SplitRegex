import torch
from collections import Counter

#pos, other['sequence']
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

