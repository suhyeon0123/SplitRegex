#!/bin/bash

python data_generator/random_data/regex_generator.py --alphabet_size 8 --is_train --number 1000000
python data_generator/random_data/data_generator.py --alphabet_size 8 --is_train
python data_generator/random_data/regex_generator.py --alphabet_size 8 --number 100000
python data_generator/random_data/data_generator.py --alphabet_size 8
