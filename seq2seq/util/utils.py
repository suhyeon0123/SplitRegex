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
        self.df = pd.read_csv(file_path, header=None)
        self.input = self.df[self.df.columns[:10]]
        self.output = self.df[self.df.columns[10:20]]
        self.regex = self.df[self.df.columns[20]]



        '''print(torch.LongTensor([list(map(int, lst)) for lst in tmp]))
        print(torch.LongTensor(list(map(lambda x: list(str(x)), list(self.input.iloc[0])))))

        print(torch.LongTensor(list(self.input.iloc[0])))
        exit()'''


        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input = self.input.iloc[idx]
        processed_list = []
        for sequence in list(input):
            listed_sequence = list(str(sequence))
            tmp = []

            for i in range(10):
                if i < len(listed_sequence):
                    tmp.append(listed_sequence[i])
                else:
                    tmp.append('<pad>')

            processed_list.append(tmp)
        numericalized_input = self.vocab.numericalize(processed_list)


        output = self.output.iloc[idx]
        processed_list = []
        for sequence in list(output):
            listed_sequence = list(str(sequence))
            tmp = []

            for i in range(10):
                if i < len(listed_sequence):
                    tmp.append(listed_sequence[i])
                else:
                    tmp.append('<pad>')

            processed_list.append(tmp)
        numericalized_output = self.vocab.numericalize(processed_list)


        return numericalized_input, numericalized_output, self.regex.iloc[idx]


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
    pad_idx = dataset.vocab.stoi['<pad>']
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