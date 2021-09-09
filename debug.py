import pickle
import os
import re2 as re
path = './log_data/practical/snort/alpharegex'
file_list = sorted(os.listdir(path), key=lambda x: re.sub('\D','',x))
print(file_list)

def membership(regex, string):
    return bool(re.fullmatch(regex, string))

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


DC_score = 0
Direct_score=0
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
print(DC_score)
print(Direct_score)


        # print('-'*50)
        # print('Positive Strings:', ', '.join(log_data['pos']))
        # print('Negative Strings:', ', '.join(log_data['neg']))
        # print('Target Regex:', ''.join(log_data['Target_string']))
        # print('-'*50)
        # print(f'{i}th Generated Regex (via DC): ' + str(log_data['DC_answer']) +  '(' + str(log_data['DC_time'] < 20) + '), Time Taken: '  + str(log_data['DC_time']))
        # print(f'{i}th Generated Regex (direct): {log_data["Direct_answer"]}, Time Taken: ' + str(log_data['Direct_time']))
        # print(f'Divide-and-conquer win rate over Direct: {log_data["win_rate"]:.4f}%, Direct Total Time: {log_data["Direct_total_time"]:.4f}, DC Total Time: {log_data["DC_total_time"]:.4f}')
        # print(f'DC Success Ratio: {log_data["DC_success_ratio"]:.4f}%, Direct Success Ratio: {log_data["Direct_success_ratio"]:.4f}%')
        # print(log_data['pos_validation'])
        # print(log_data['neg_validation'])
        # print('-'*50)

