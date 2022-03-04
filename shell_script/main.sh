#!/bin/bash


# alpharegex & practical data
python main.py --data_path ./data/practical_data/integrated/test_regexlib.csv --log_path ./log_data/regexlib --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex
python main.py --data_path ./data/practical_data/integrated/test_snort.csv --log_path ./log_data/snort --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex
python main.py --data_path ./data/practical_data/integrated/test_practicalregex.csv --log_path ./log_data/practicalregex --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex

# alpharegex & random data
python main.py --data_path ./data/random_data/size_2/test.csv --log_path ./log_data/random2 --checkpoint_pos ./saved_models/random/gru__256__2__2/best_accuracy --data_type random --sub_model alpharegex --alphabet_size 2
python main.py --data_path ./data/random_data/size_10/test.csv --log_path ./log_data/random10 --checkpoint_pos ./saved_models/random/gru__256__2__2/best_accuracy --data_type random --sub_model alpharegex --alphabet_size 10


