from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch

class Vocabulary:
    def __init__(self):
        self.itos = {11: '<pad>', 12: '<unk>'}
        self.itos.update({i: str(i) for i in range(10)})

        self.stoi = {'<pad>': 11, '<unk>': 12}
        self.stoi.update({str(i): i for i in range(10)})

    def __len__(self):
        return len(self.itos)

    def numericalize(self, text):
        return [self.stoi[token] for token in list(text)]


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, header=None)
        self.input = self.df[self.df.columns[:10]]

        ## certain d
        print(self.input.shape)
        print(type(self.input))
        print(list(self.input.iloc[0]))

        processed_list = []
        for sequence in list(self.input.iloc[0]):
            listed_sequence = list(str(sequence))

            tmp = []

            for i in range(10):
                if i < len(listed_sequence):
                    tmp.append(int(listed_sequence[i]))
                else:
                    tmp.append(int(listed_sequence[i]))



            processed_list.append(i)

        print(list(map(lambda x:list(str(x)),list(self.input.iloc[0]))))
        lst = list(map(lambda x: list(str(x)), list(self.input.iloc[0])))

        print(torch.LongTensor([list(map(int, lst)) for lst in tmp]))
        print(torch.LongTensor(list(map(lambda x: list(str(x)), list(self.input.iloc[0])))))

        print(torch.LongTensor(list(self.input.iloc[0])))
        exit()
        self.output = self.df[self.df.columns[10:20]]
        self.regex = self.df[self.df.columns[20]]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.input.iloc[idx], self.output.iloc[idx], self.regex.iloc[idx]

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_idx)
        outputs = [item[1] for item in batch]
        outputs = pad_sequence(outputs, batch_first=True, padding_value=self.pad_idx)

        return inputs, outputs

def get_loader(file_path, batch_size, num_worker=8, shuffle = True):
    dataset = CustomDataset(file_path)
    pad_idx = dataset.vocab.stoi['<pad>']
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_worker,
                        shuffle=shuffle, collate_fn=MyCollate(pad_idx=pad_idx))
    return loader


