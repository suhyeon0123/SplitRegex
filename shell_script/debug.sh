#!/bin/bash



# # Table 1, 2

# alpharegex
python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 3 --num 3000 
python debug.py --path log_data/snort/sequential/alpharegex --time_limit 3 --num 3000 
python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 3 --num 3000 

# blue_fringe
python debug.py --path log_data/regexlib/sequential/blue_fringe --time_limit 3 --num 3000
python debug.py --path log_data/snort/sequential/blue_fringe --time_limit 3 --num 3000
python debug.py --path log_data/practicalregex/sequential/blue_fringe --time_limit 3 --num 3000



# Figure 2
# python debug.py --path log_data/random/2/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/4/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/6/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/8/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 3 --num 1000 



# python debug_total.py --path log_data/regexlib/sequential/second/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/snort/sequential/second/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/practicalregex/sequential/second/alpharegex --time_limit 3 --num 3000

# python debug_total.py --path log_data/regexlib/sequential/third/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/snort/sequential/third/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/practicalregex/sequential/third/alpharegex --time_limit 3 --num 3000



# Table 3
# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 1 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 1 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 1 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 1 --num 3000 --exclude_GT

# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 3 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT

# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 5 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 5 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 5 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 5 --num 3000 --exclude_GT

# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 10 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 10 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 10 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 10 --num 3000 --exclude_GT



# Table 4
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT

# python debug.py --path log_data/regexlib/prefix/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/prefix/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/prefix/alpharegex --time_limit 3 --num 3000 --exclude_GT

# python debug.py --path log_data/regexlib/parallel/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/parallel/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/parallel/alpharegex --time_limit 3 --num 3000 --exclude_GT



# # A.3
# python debug.py --path log_data/regex_perturb/alpharegex --time_limit 3


# # A.4
# python debug.py --path log_data/regexlib/sequential/regex_generator --time_limit 15 --num 3000
# python debug.py --path log_data/snort/sequential/regex_generator --time_limit 15 --num 3000
# python debug.py --path log_data/practicalregex/sequential/regex_generator --time_limit 15 --num 3000


# # A.5
# python debug.py --path log_data/KB13_full/alpharegex --time_limit 3 
# python debug.py --path log_data/NL-RX-Turk_full/alpharegex --time_limit 3

