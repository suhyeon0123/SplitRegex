#!/bin/bash




# blue_fringe & practical data
python main.py --data_path ./data/practical_data/test_regexlib.csv --log_path ./log_data/regexlib --checkpoint_pos ./saved_models/practical/lstm__256__2__2/best_accuracy --data_type practical --sub_model blue_fringe
python main.py --data_path ./data/practical_data/test_snort.csv --log_path ./log_data/snort --checkpoint_pos ./saved_models/practical/lstm__256__2__2/best_accuracy --data_type practical --sub_model blue_fringe
python main.py --data_path ./data/practical_data/test_practicalregex.csv --log_path ./log_data/practicalregex --checkpoint_pos ./saved_models/practical/lstm__256__2__2/best_accuracy --data_type practical --sub_model blue_fringe

# alpharegex & practical data
python main.py --data_path ./data/practical_data/test_regexlib.csv --log_path ./log_data/regexlib --checkpoint_pos ./saved_models/practical/lstm__256__2__2/best_accuracy --data_type practical --sub_model alpharegex
python main.py --data_path ./data/practical_data/test_snort.csv --log_path ./log_data/snort --checkpoint_pos ./saved_models/practical/lstm__256__2__2/best_accuracy --data_type practical --sub_model alpharegex
python main.py --data_path ./data/practical_data/test_practicalregex.csv --log_path ./log_data/practicalregex --checkpoint_pos ./saved_models/practical/lstm__256__2__2/best_accuracy --data_type practical --sub_model alpharegex


# blue_fringe & random data
python main.py --data_path ./data/random_data/size_2/test.csv --log_path ./log_data/random2 --checkpoint_pos ./saved_models/random/lstm__128__2__2/best_accuracy --data_type random --sub_model blue_fringe --alphabet_size 2
python main.py --data_path ./data/random_data/size_10/test.csv --log_path ./log_data/random10 --checkpoint_pos ./saved_models/random/lstm__128__2__2/best_accuracy --data_type random --sub_model blue_fringe --alphabet_size 10

# alpharegex & random data
python main.py --data_path ./data/random_data/size_2/test.csv --log_path ./log_data/random2 --checkpoint_pos ./saved_models/random/lstm__128__2__2/best_accuracy --data_type random --sub_model alpharegex --alphabet_size 2
python main.py --data_path ./data/random_data/size_10/test.csv --log_path ./log_data/random10 --checkpoint_pos ./saved_models/random/lstm__128__2__2/best_accuracy --data_type random --sub_model alpharegex --alphabet_size 10


## blue_fringe & alpharegex data
#python main.py --data_path ./data/alpharegex_data/test.csv --log_path ./log_data/alpha --checkpoint_pos ./saved_models/random/lstm__128__2__2/best_accuracy --data_type random --sub_model blue_fringe --alphabet_size 2
#
## alpharegex & alpharegex data
#python main.py --data_path ./data/alpharegex_data/test.csv --log_path ./log_data/alpha --checkpoint_pos ./saved_models/random/lstm__128__2__2/best_accuracy --data_type random --sub_model alpharegex --alphabet_size 2
#



