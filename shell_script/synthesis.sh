#!/bin/bash


# Practical dataset ----------------------
python synthesis.py --data_path ./data/practical_data/test/test_regexlib.csv --log_path ./log_data/regexlib/sequential --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3
python synthesis.py --data_path ./data/practical_data/test/test_snort.csv --log_path ./log_data/snort/sequential --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3
python synthesis.py --data_path ./data/practical_data/test/test_practicalregex.csv --log_path ./log_data/practicalregex/sequential --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3

python synthesis.py --data_path ./data/practical_data/test/test_regexlib.csv --log_path ./log_data/regexlib/sequential --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model blue_fringe --time_limit 3
python synthesis.py --data_path ./data/practical_data/test/test_snort.csv --log_path ./log_data/snort/sequential --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model blue_fringe --time_limit 3
python synthesis.py --data_path ./data/practical_data/test/test_practicalregex.csv --log_path ./log_data/practicalregex/sequential --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model blue_fringe --time_limit 3


# Random dataset ------------------------

python synthesis.py --data_path ./data/random_data/size_2/test.csv --log_path ./log_data/random/2/sequential --checkpoint_pos ./saved_models/random/gru__256__2__2 --data_type random --sub_model alpharegex --alphabet_size 2 --time_limit 3
python synthesis.py --data_path ./data/random_data/size_4/test.csv --log_path ./log_data/random/4/sequential --checkpoint_pos ./saved_models/random/gru__256__2__2 --data_type random --sub_model alpharegex --alphabet_size 4 --time_limit 3
python synthesis.py --data_path ./data/random_data/size_6/test.csv --log_path ./log_data/random/6/sequential --checkpoint_pos ./saved_models/random/gru__256__2__2 --data_type random --sub_model alpharegex --alphabet_size 6 --time_limit 3
python synthesis.py --data_path ./data/random_data/size_8/test.csv --log_path ./log_data/random/8/sequential --checkpoint_pos ./saved_models/random/gru__256__2__2 --data_type random --sub_model alpharegex --alphabet_size 8 --time_limit 3
python synthesis.py --data_path ./data/random_data/size_10/test.csv --log_path ./log_data/random/10/sequential --checkpoint_pos ./saved_models/random/gru__256__2__2 --data_type random --sub_model alpharegex --alphabet_size 10 --time_limit 3


# Other synthesis methods
# python synthesis.py --data_path ./data/practical_data/test/test_regexlib.csv --log_path ./log_data/regexlib/parallel --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3 --synthesis_strategy parallel --exclude_GT --exclude_Direct
# python synthesis.py --data_path ./data/practical_data/test/test_snort.csv --log_path ./log_data/snort/parallel --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3 --synthesis_strategy parallel --exclude_GT --exclude_Direct
# python synthesis.py --data_path ./data/practical_data/test/test_practicalregex.csv --log_path ./log_data/practicalregex/parallel --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3 --synthesis_strategy parallel --exclude_GT --exclude_Direct

# python synthesis.py --data_path ./data/practical_data/test/test_regexlib.csv --log_path ./log_data/regexlib/prefix --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3 --synthesis_strategy sequential_prefix --exclude_GT --exclude_Direct
# python synthesis.py --data_path ./data/practical_data/test/test_snort.csv --log_path ./log_data/snort/prefix --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3 --synthesis_strategy sequential_prefix --exclude_GT --exclude_Direct
# python synthesis.py --data_path ./data/practical_data/test/test_practicalregex.csv --log_path ./log_data/practicalregex/prefix --checkpoint_pos ./saved_models/practical/gru__256__2__2 --data_type practical --sub_model alpharegex --time_limit 3 --synthesis_strategy sequential_prefix --exclude_GT --exclude_Direct

