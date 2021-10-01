import pickle
import os
import re2 as re
from FAdo.fa import *
from FAdo.cfg import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', dest='path', help='Path to experiment result directory.')
opt = parser.parse_args()


path = opt.path
# path = './log_data/random42/blue_fringe'
file_list = sorted(os.listdir(path), key=lambda x: int(re.sub('\D*','',x)))

print(path)
print('length of data:', str(len(file_list)))

def membership(regex, string):
    return bool(re.fullmatch(regex, string))

def membership2(regex, string):
    return reex.str2regexp(regex).evalWordP(string)

if 'blue_fringe' in path :
    membership = membership2



with open(path + '/' + file_list[-1], 'rb') as fr:
    log_data = pickle.load(fr)
    print('-' * 50)
    print('Positive Strings:', ', '.join(log_data['pos']))
    print('Negative Strings:', ', '.join(log_data['neg']))
    print('Target Regex:', ''.join(log_data['Target_string']))
    print('-' * 50)
    print(f'{file_list[-1]}th Generated Regex (via DC): ' + str(log_data['DC_answer']) + '(' + str(
        log_data['DC_time'] < 20) + '), Time Taken: ' + str(log_data['DC_time']))
    print(f'{file_list[-1]}th Generated Regex (direct): {log_data["Direct_answer"]}, Time Taken: ' + str(log_data['Direct_time']))
    print(
        f'Divide-and-conquer win rate over Direct: {log_data["win_rate"]:.4f}%, Direct Total Time: {log_data["Direct_total_time"]:.4f}, DC Total Time: {log_data["DC_total_time"]:.4f}')
    print(
        f'DC Success Ratio: {log_data["DC_success_ratio"]:.4f}%, Direct Success Ratio: {log_data["Direct_success_ratio"]:.4f}%')
    print(log_data['pos_validation'])
    print(log_data['neg_validation'])
    print('-' * 50)

    DC_success_rate = log_data["DC_success_ratio"]
    Direct_success_rate = log_data["Direct_success_ratio"]


DC_score = 0
Direct_score = 0


for file_name in file_list:
    with open(path + '/' + file_name, 'rb') as fr:
        log_data = pickle.load(fr)


        if log_data['DC_time'] <3:
            # DC scoring
            for string in log_data['pos_validation']:
                if membership(log_data['DC_answer'], string):
                    DC_score += 1

            for string in log_data['neg_validation']:
                if not membership(log_data['DC_answer'], string):
                    DC_score += 1

        if log_data['Direct_time'] < 3:
            # Direct scoring
            for string in log_data['pos_validation']:
                if membership(log_data['Direct_answer'], string):
                    Direct_score += 1
                    

            for string in log_data['neg_validation']:
                if not membership(log_data['Direct_answer'], string):
                    Direct_score += 1

print('DC success ratio : ', DC_success_rate)          
print('Direct success ratio : ', Direct_success_rate)          
print('DC score : ', str(DC_score/20/len(file_list)))
print('Direct score : ', str(Direct_score/20/len(file_list)))
print()
