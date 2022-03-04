#!/bin/bash

seed=152

# python practice_seed.py --seed $seed
python train.py --train_path ./data/practical_data/integrated/train.csv --valid_path ./data/practical_data/integrated/valid.csv --expt_dir saved_models/practical --gru --hidden_size 256 --num_layer 2 --bidirectional --batch_size 512 --dropout_en 0.4 --dropout_de 0.4 --weight_decay 0.000001 --add_seed $seed