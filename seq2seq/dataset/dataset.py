from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


class Vocabulary:
    def __init__(self):
        self.itos = []
        # 0-9
        for i in range(10):
            self.itos.append(str(i))
        # A-Z
        for i in range(65,91):
            self.itos.append(chr(i))
        # a-z
        for i in range(97, 123):
            self.itos.append(chr(i))

        self.itos += ['!', '<pad>', '<unk>']

        self.stoi = dict((x, i) for i, x in enumerate(self.itos))

    def __len__(self):
        return len(self.itos)

    def get_idx(self,text):
        tmp = self.stoi.get(text)
        if tmp is None:
            tmp = 62
        return tmp

    def text2idx(self, text):
        return list(map(self.get_idx, text))


NUM_EXAMPLES = 10
MAX_SEQUENCE_LENGTH = 50


class CustomDataset(Dataset):
    POS_COL = 0
    NEG_COL = POS_COL + NUM_EXAMPLES
    LABEL_COL = NEG_COL + NUM_EXAMPLES
    REGEX_COL = LABEL_COL + NUM_EXAMPLES

    def __init__(self, file_path, object='train'):
        if object == 'train':
            self.df = pd.read_csv(file_path, header=None, dtype=str, na_filter=False)
            self.df = self.df.head(int(len(self.df)*0.8))
            self.input = self.df[self.df.columns[CustomDataset.POS_COL:CustomDataset.NEG_COL]]
            self.output = self.df[self.df.columns[CustomDataset.LABEL_COL:CustomDataset.REGEX_COL]]
            self.regex = self.df[self.df.columns[CustomDataset.REGEX_COL]]
        elif object == 'valid':
            self.df = pd.read_csv(file_path, header=None, dtype=str, na_filter=False)
            self.df = self.df.head(-int(len(self.df) * 0.8))
            self.input = self.df[self.df.columns[CustomDataset.POS_COL:CustomDataset.NEG_COL]]
            self.output = self.df[self.df.columns[CustomDataset.LABEL_COL:CustomDataset.REGEX_COL]]
            self.regex = self.df[self.df.columns[CustomDataset.REGEX_COL]]
        else:
            self.df = pd.read_csv(file_path, header=None, dtype=str, na_filter=False)
            self.input = self.df[self.df.columns[CustomDataset.POS_COL:CustomDataset.NEG_COL]]
            self.output = self.df[self.df.columns[CustomDataset.NEG_COL:CustomDataset.LABEL_COL]]
            self.regex = self.df[self.df.columns[CustomDataset.REGEX_COL]]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    def _translate_sequences(self, sequences):
        processed_list = []
        for sequence in map(str.strip, sequences):
            if sequence == '<pad>':
                tmp = ['<pad>'] * MAX_SEQUENCE_LENGTH
            else:
                tmp = list(sequence) + ['<pad>'] * (MAX_SEQUENCE_LENGTH - len(sequence))
            processed_list.append(self.vocab.text2idx(tmp[:MAX_SEQUENCE_LENGTH]))

        return processed_list

    def __getitem__(self, idx):
        return (self._translate_sequences(self.input.iloc[idx]),
                self._translate_sequences(self.output.iloc[idx]),
                self.regex.iloc[idx])


def get_loader(file_path, batch_size, object, num_worker=0, shuffle=True):

    dataset = CustomDataset(file_path, object)
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
        return list(map(lambda x: x[6:], saved_decomposed_regex))
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
