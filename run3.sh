#!/bin/bash

python data_generator/random_data/data_generator.py --alphabet_size 2 --is_train
python data_generator/random_data/data_generator.py --alphabet_size 2

python data_generator/random_data/data_generator.py --alphabet_size 4 --is_train
python data_generator/random_data/data_generator.py --alphabet_size 4

python data_generator/random_data/data_generator.py --alphabet_size 6 --is_train
python data_generator/random_data/data_generator.py --alphabet_size 6

python data_generator/random_data/data_generator.py --alphabet_size 8 --is_train
python data_generator/random_data/data_generator.py --alphabet_size 8

python data_generator/random_data/data_generator.py --alphabet_size 10 --is_train
python data_generator/random_data/data_generator.py --alphabet_size 10
