#!/bin/bash

## practical data train
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --gru --hidden_size 64 --num_layer 1
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --gru --hidden_size 64 --num_layer 1 --bidirectional
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --gru --hidden_size 64 --num_layer 2 --bidirectional
python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --gru --hidden_size 128 --num_layer 2 --bidirectional
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --gru --hidden_size 256 --num_layer 2 --bidirectional
#
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --hidden_size 64 --num_layer 1
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --hidden_size 64 --num_layer 1 --bidirectional
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --hidden_size 64 --num_layer 2 --bidirectional
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --hidden_size 128 --num_layer 2 --bidirectional
#python train.py --train_path ./data/practical_data/train.csv --expt_dir ./saved_models/practical --hidden_size 256 --num_layer 2 --bidirectional
#
#
## random data train
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --gru --hidden_size 64 --num_layer 1
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --gru --hidden_size 64 --num_layer 1 --bidirectional
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --gru --hidden_size 64 --num_layer 2 --bidirectional
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --gru --hidden_size 128 --num_layer 2 --bidirectional
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --gru --hidden_size 256 --num_layer 2 --bidirectional
#
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --hidden_size 64 --num_layer 1
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --hidden_size 64 --num_layer 1 --bidirectional
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --hidden_size 64 --num_layer 2 --bidirectional
python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --hidden_size 128 --num_layer 2 --bidirectional
#python train.py --train_path ./data/random_data/train.csv --expt_dir ./saved_models/random --hidden_size 256 --num_layer 2 --bidirectional


