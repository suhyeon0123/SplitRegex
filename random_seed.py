import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--seed', action='store', dest='seed',
                    help='seed', type=int, default=1)


opt = parser.parse_args()

train=pd.read_csv('./data/random_data/integrated/mega_train.csv',header=None)
valid=pd.read_csv('./data/random_data/integrated/mega_valid.csv',header=None)

data=pd.concat([train, valid])

data=data.sample(frac=1)

pathlib.Path('./data/random_data/integrated/' +  str(opt.seed)).mkdir(parents=True, exist_ok=True)
data.head(int(len(data)*0.9)).to_csv('./data/random_data/integrated/' +  str(opt.seed)  + '/mega_train.csv', sep=',', index = False, header=False)
data.tail(int(len(data)*0.1)).to_csv('./data/random_data/integrated/' +  str(opt.seed)  + '/mega_valid.csv', sep=',', index = False, header=False)
