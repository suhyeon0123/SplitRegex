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
        return list(map(lambda x: [torch.tensor(self.stoi[token]) for token in x], text))


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, header=None, dtype=str)
        self.input = self.df[self.df.columns[:10]]
        self.output = self.df[self.df.columns[10:20]]
        self.regex = self.df[self.df.columns[20]]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        items = [self.input.iloc[idx], self.output.iloc[idx]]
        numericalized_items = []

        for item_idx, item in enumerate(items):
            processed_list = []
            for ix, sequence in enumerate(list(item)):
                if ix == 0 and item_idx == 0:
                    listed_sequence = list(str(sequence))
                else:
                    listed_sequence = list(str(sequence[1:]))
                tmp = []

                for i in range(10):
                    if i < len(listed_sequence):
                        tmp.append(listed_sequence[i])
                    else:
                        tmp.append('<pad>')

                processed_list.append(tmp)

            numericalized_items.append(self.vocab.numericalize(processed_list))

        return numericalized_items[0], numericalized_items[1], self.regex.iloc[idx]


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):

        inputs = [item[0] for item in batch]
        for set_idx in range(10):
            inputs[set_idx] = pad_sequence(inputs[set_idx], batch_first=True, padding_value=self.pad_idx)

        outputs = torch.tensor([item[1] for item in batch])
        for set_idx in range(10):
            outputs[set_idx] = pad_sequence(outputs[set_idx], batch_first=True, padding_value=self.pad_idx)

        return inputs, outputs


def get_loader(file_path, batch_size, num_worker=0, shuffle=True):
    dataset = CustomDataset(file_path)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_worker,
                        shuffle=shuffle, pin_memory=True)
    return loader


def decomposing_regex(regex):
    saved_decomposed_regex = []
    bracket = 0
    for letter in regex[1:]:
        if letter == '(':
            if bracket == 0:
                saved_decomposed_regex.append('')
            else:
                saved_decomposed_regex[-1] = saved_decomposed_regex[-1] + letter
            bracket += 1
        elif letter == ')':
            if bracket != 1:
                saved_decomposed_regex[-1] = saved_decomposed_regex[-1] + letter
            bracket -= 1
        else:
            saved_decomposed_regex[-1] = saved_decomposed_regex[-1] + letter

    return list(map(lambda x: x[6:], saved_decomposed_regex))


def batch_preprocess(inputs, outputs, regex):
    for batch_idx in range(len(inputs)):
        inputs[batch_idx] = torch.stack(inputs[batch_idx], dim=0)
        outputs[batch_idx] = torch.stack(outputs[batch_idx], dim=0)

    inputs = torch.stack(inputs, dim=0)
    outputs = torch.stack(outputs, dim=0)

    inputs = inputs.permute(2, 0, 1)
    outputs = outputs.permute(2, 0, 1)

    regex = list(map(lambda x: decomposing_regex(x), regex))

    return inputs, outputs, regex
