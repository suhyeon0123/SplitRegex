from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import re2 as re


class Vocabulary:
    def __init__(self):
        self.itos = []
        # 0-9
        for i in range(10):
            self.itos.append(str(i))
        # A-Z
        for i in range(65, 91):
            self.itos.append(chr(i))
        # a-z
        for i in range(97, 123):
            self.itos.append(chr(i))

        self.itos += ['!', '_', '<pad>', '<unk>']

        self.stoi = dict((x, i) for i, x in enumerate(self.itos))

    def __len__(self):
        return len(self.itos)

    def get_idx(self, text):
        tmp = self.stoi.get(text)
        if tmp is None:
            tmp = self.stoi.get('!')
        return tmp

    def text2idx(self, text):
        return list(map(self.get_idx, text))


NUM_EXAMPLES = 10

POS_IDX = 0
VALID_POS_IDX = 10
NEG_IDX = 20
VALID_NEG_IDX = 30
TAG_IDX = 40
VALID_TAG_IDX = 50
REGEX_IDX = 60

class CustomDataset(Dataset):

    def __init__(self, file_path, object='train', max_len=10):
        if object == 'test':
            self.test = True
        else:
            self.test = False

        self.df = pd.read_csv(file_path, header=None, dtype=str, na_filter=False)
        self.MAX_SEQUENCE_LENGTH = max_len

        self.pos = self.df[self.df.columns[POS_IDX:POS_IDX+NUM_EXAMPLES]]
        self.valid_pos = self.df[self.df.columns[VALID_POS_IDX:VALID_POS_IDX+NUM_EXAMPLES]]
        self.neg = self.df[self.df.columns[NEG_IDX:NEG_IDX+NUM_EXAMPLES]]
        self.valid_neg = self.df[self.df.columns[VALID_NEG_IDX:VALID_NEG_IDX+NUM_EXAMPLES]]
        self.tag = self.df[self.df.columns[TAG_IDX:TAG_IDX+NUM_EXAMPLES]]
        self.regex = self.df[self.df.columns[REGEX_IDX]]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    def _translate_sequences(self, sequences):
        processed_list = []
        for sequence in map(str.strip, sequences):
            if sequence == '<pad>':
                tmp = ['<pad>'] * self.MAX_SEQUENCE_LENGTH
            else:
                tmp = list(sequence) + ['<pad>'] * (self.MAX_SEQUENCE_LENGTH - len(sequence))
            processed_list.append(self.vocab.text2idx(tmp[:self.MAX_SEQUENCE_LENGTH]))

        return processed_list

    def __getitem__(self, idx):
        if self.test:
            return (self._translate_sequences(self.pos.iloc[idx]),
                    self._translate_sequences(self.neg.iloc[idx]),
                    self.regex.iloc[idx],
                    self._translate_sequences(self.valid_pos.iloc[idx]),
                    self._translate_sequences(self.valid_neg.iloc[idx]))
        else:
            return (self._translate_sequences(self.pos.iloc[idx]),
                    self._translate_sequences(self.tag.iloc[idx]),
                    self.regex.iloc[idx])


def get_loader(file_path, batch_size, object, num_worker=0, shuffle=True, max_len=10):
    dataset = CustomDataset(file_path, object, max_len=max_len)
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

    if '(?P<t' in regex:
        return list(map(lambda x: re.sub('\?P\<t\d*?\>', '', x), saved_decomposed_regex))
    else:
        return list(map(lambda x: x[0:], saved_decomposed_regex))


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
