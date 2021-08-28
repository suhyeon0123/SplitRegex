from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch


class Vocabulary:
    def __init__(self):
        # self.itos = [
        #         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        #         '<pad>', '<unk>']
        self.itos = [
                '<pad>', '<unk>']
        self.stoi = dict((x, i) for i, x in enumerate(self.itos))

    def __len__(self):
        return len(self.itos)

    def text2idx(self, text):
        return list(map(self.stoi.get, text))


NUM_EXAMPLES = 10
MAX_SEQUENCE_LENGTH = 10


class CustomDataset(Dataset):
    INPUT_COL = 0
    OUTPUT_COL = INPUT_COL + NUM_EXAMPLES
    REGEX_COL = OUTPUT_COL + NUM_EXAMPLES

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, header=None, dtype=str)
        self.input = self.df[self.df.columns[CustomDataset.INPUT_COL:CustomDataset.OUTPUT_COL]]
        self.output = self.df[self.df.columns[CustomDataset.OUTPUT_COL:CustomDataset.REGEX_COL]]
        self.regex = self.df[self.df.columns[CustomDataset.REGEX_COL]]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    def _translate_sequences(self, sequences):
        processed_list = []
        for sequence in map(str.strip, sequences):
            tmp = list(sequence) + ['<pad>'] * (MAX_SEQUENCE_LENGTH - len(sequence))
            processed_list.append(self.vocab.text2idx(tmp[:MAX_SEQUENCE_LENGTH]))
        return processed_list

    def __getitem__(self, idx):

        return (self._translate_sequences(self.input.iloc[idx]),
                self._translate_sequences(self.output.iloc[idx]),
                self.regex.iloc[idx])


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
