import random
import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
random.seed(int(config['seed']['integrate_data']))

data_list = ['./data/random_data/size_2', './data/random_data/size_4', './data/random_data/size_6', './data/random_data/size_8', './data/random_data/size_10']
save_file = open('./data/random_data/train.csv', 'w+')


data = []
for path in data_list:
    file = open(path + '/train.csv', 'r')
    data += file.readlines()

random.shuffle(data)

for line in data:
    save_file.write(line)



