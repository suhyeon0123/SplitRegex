import random
import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
random.seed(int(config['seed']['integrate_data']))

data_list = ['./data/practical_data/org/snort-clean.csv', './data/practical_data/org/regexlib-clean.csv', './data/practical_data/org/practical_regexes.csv']

train_file = open('./data/practical_data/integrated/train.csv', 'w+')
valid_file = open('./data/practical_data/integrated/valid.csv', 'w+')
test_snort_file = open('data/practical_data/integrated/test_snort.csv', 'w')
test_regexlib_file = open('data/practical_data/integrated/test_regexlib.csv', 'w')
test_practical_file = open('data/practical_data/integrated/test_practicalregex.csv', 'w')


intergrated_data = []

for data_idx, path in enumerate(data_list):
    file = open(path, 'r')
    data = file.readlines()
    random.shuffle(data)


    for line in data[len(data) - 10000:]:
        if data_idx == 0:
            test_snort_file.write(line)   
        elif data_idx == 1:
            test_regexlib_file.write(line)   
        else:
            test_practical_file.write(line)   
    intergrated_data += data[:len(data)-10000]


random.shuffle(intergrated_data)
for line in intergrated_data[:int(len(intergrated_data)*0.9)]:
    train_file.write(line)
for line in intergrated_data[int(len(intergrated_data)*0.9):]:
    valid_file.write(line)




