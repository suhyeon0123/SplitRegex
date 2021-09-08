import pickle


for i in range(10):
    with open('log_data/practical_data/' + str(i) + '.pickle', 'rb') as fr:
        log_data = pickle.load(fr)
        print('-'*50)
        print('Positive Strings:', ', '.join(log_data['pos']))
        print('Negative Strings:', ', '.join(log_data['neg']))
        print('Target Regex:', ''.join(log_data['Target_string']))
        print('-'*50)
        print(f'{i}th Generated Regex (via DC): ' + str(log_data['DC_answer']) +  '(' + str(log_data['DC_time'] < 20) + '), Time Taken: '  + str(log_data['DC_time']))
        print(f'{i}th Generated Regex (direct): {log_data["Direct_answer"]}, Time Taken: ' + str(log_data['Direct_time']))
        print(f'Divide-and-conquer win rate over Direct: {log_data["win_rate"]:.4f}%, Direct Total Time: {log_data["Direct_total_time"]:.4f}, DC Total Time: {log_data["DC_total_time"]:.4f}')
        print(f'DC Success Ratio: {log_data["DC_success_ratio"]:.4f}%, Direct Success Ratio: {log_data["Direct_success_ratio"]:.4f}%')
        print(log_data['pos_validation'])
        print(log_data['neg_validation'])
        print('-'*50)

