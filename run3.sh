#!/bin/bash

python data_generator/random_data/regex_generator.py --alphabet_size 6 --is_train --number 1000000
python data_generator/random_data/data_generator.py --alphabet_size 6 --is_train
python data_generator/random_data/regex_generator.py --alphabet_size 6 --number 100000
python data_generator/random_data/data_generator.py --alphabet_size 6
