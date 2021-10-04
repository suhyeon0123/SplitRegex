#!/bin/bash

seed=152

# python practice_seed.py --seed $seed
python model_trainer/train.py --train_path ./data/practical_data/integrated/$seed/mega_train.csv --valid_path ./data/practical_data/integrated/$seed/mega_valid.csv --expt_dir saved_models/practical/$seed --gru --hidden_size 256 --num_layer 2 --bidirectional --batch_size 512 --dropout_en 0.4 --dropout_de 0.4 --weight_decay 0.000001 --add_seed $seed