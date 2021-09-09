# file = open('./data/practical_data/train.csv', 'r')
# data = file.readlines()
# print(len(data))
#
# file = open('./data/practical_data/test_practicalregex.csv', 'r')
# data = file.readlines()
# print(len(data))
#
# file = open('./data/practical_data/test_regexlib.csv', 'r')
# data = file.readlines()
# print(len(data))
#
# file = open('./data/practical_data/test_snort.csv', 'r')
# data = file.readlines()
# print(len(data))
#
# exit()


data_list = ['./data/random_data/size_2', './data/random_data/size_4', './data/random_data/size_6', './data/random_data/size_8', './data/random_data/size_10']

for path in data_list:
    file = open(path + '/train.csv', 'r')
    mini_file = open(path + '/train_mini.csv', 'w')
    data = file.readlines()[:20000]
    for idx, line in enumerate(data):
        mini_file.write(line)

for path in data_list:
    file = open(path + '/test.csv', 'r')
    mini_file = open(path + '/test_mini.csv', 'w')
    data = file.readlines()[:1000]
    for idx, line in enumerate(data):
        mini_file.write(line)




