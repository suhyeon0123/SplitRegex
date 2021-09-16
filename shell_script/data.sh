#!/bin/bash

python data_generator/practical_data/data_generator.py

python data_generator/random_data/data_generator.py --alphabet_size 2 --is_train --number 40000
python data_generator/random_data/data_generator.py --alphabet_size 2 --number 1000

python data_generator/random_data/data_generator.py --alphabet_size 4 --is_train --number 40000
python data_generator/random_data/data_generator.py --alphabet_size 4 --number 1000

python data_generator/random_data/data_generator.py --alphabet_size 6 --is_train --number 40000
python data_generator/random_data/data_generator.py --alphabet_size 6 --number 1000

python data_generator/random_data/data_generator.py --alphabet_size 8 --is_train --number 40000
python data_generator/random_data/data_generator.py --alphabet_size 8 --number 1000

python data_generator/random_data/data_generator.py --alphabet_size 10 --is_train --number 40000
python data_generator/random_data/data_generator.py --alphabet_size 10 --number 1000

python data_generator/random_data/data_integration.py