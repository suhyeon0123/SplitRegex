import os, sys
import re2 as re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'submodels', 'SoftConciseNormalForm')))
from parsetree import *
from xeger import Xeger
import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('--alphabet_size', action='store', dest='alphabet_size',
                    help='define the alphabet size of the regex', type=int, default=5)
parser.add_argument('--is_train', action='store_true', dest='is_train',
                    help='Indicate if the data is used for train or test', default=False)

opt = parser.parse_args()


MAX_SEQUENCE_LENGTH = 10
EXAMPLE_NUM = 20


def generate_data():
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    random.seed(int(config['seed']['random_data'])+ int(opt.alphabet_size))
    xeger = Xeger(limit=5)
    xeger.seed(int(config['seed']['random_data'])+ int(opt.alphabet_size))

    if opt.is_train:
        DATA_USAGE = 'train'
    else:
        DATA_USAGE = 'test'

    regex_file = open('./data/random_data/size_' + str(opt.alphabet_size) + '/' + DATA_USAGE + '_regex', 'r')
    # save_file = open('./data/random_data/size_' + str(opt.alphabet_size) + '/' + DATA_USAGE + '.csv', 'w')
    regexes = [x.strip() for x in regex_file.readlines()]

    data_num = 0
    max_len = 0
    for idx, regex in enumerate(regexes):
        raw = regex
        # pos examples 생성
        pos = []

        for i in range(200):
            example = xeger.xeger(regex).strip("'")
            if 0 < len(example) <= MAX_SEQUENCE_LENGTH and example not in pos:
                pos.append(example)
            if len(pos) == EXAMPLE_NUM:
                break

        if len(pos) != EXAMPLE_NUM:
            continue


        # neg examples 생성
        neg = []
        for _ in range(1000):

            # random regex생성
            str_list = []
            for j in range(random.randrange(1, MAX_SEQUENCE_LENGTH+1)):
                str_list.append(str(random.randrange(0, opt.alphabet_size)))
            tmp = ''.join(str_list)

            # random regex가 맞지 않다면 추가
            if not bool(re.fullmatch(regex, tmp)) and tmp not in neg:
                neg.append(tmp)

            if len(neg) == EXAMPLE_NUM:
                break

        if not len(neg) == EXAMPLE_NUM:
            continue


        # Tag 전처리
        str_list = []
        bracket = 0
        tagIndex = 1
        for letter in regex:
            str_list.append(letter)

            if letter == '(':
                if bracket == 0:
                    str_list.append("?P<t" + str(tagIndex) + ">")
                    tagIndex += 1
                bracket += 1
            elif letter == ')':
                bracket -= 1
        regex = "".join(str_list)


        subregex_list = []
        bracket = 0
        for letter in regex:
            if letter == '(':
                if bracket == 0:
                    subregex_list.append('')
                else:
                    subregex_list[-1] = subregex_list[-1] + letter
                bracket += 1
            elif letter == ')':
                if bracket != 1:
                    subregex_list[-1] = subregex_list[-1] + letter
                bracket -= 1
            else:
                subregex_list[-1] = subregex_list[-1] + letter

        SIGMA_STAR = '0'
        # print()
        # print(regex)

        # templetes 생성
        templete = []
        for example in pos:
            str_list = []
            dic = re.fullmatch(regex, example).groupdict()
            label_num = 1
            for i in range(1, len(dic) + 1):

                targetstring = dic['t'+str(i)]
                targetregex = re.sub('\?P\<t\d*?\>' , '', subregex_list[i-1])

                if targetregex == str(KleenStar(Or(*[Character(str(x)) for x in range(opt.alphabet_size)]))):
                    label = SIGMA_STAR
                else:
                    label = str(label_num)
                label_num += 1

                count = len(targetstring)

                for _ in range(count):
                    str_list.append(label)
            templete.append("".join(str_list))


        # save as csv file
        result = ''
        for i in range(EXAMPLE_NUM):
            if len(pos) > i:
                result += pos[i] + ', '
            else:
                result += '<pad>' + ', '

        for i in range(EXAMPLE_NUM):
            if len(templete) > i:
                result += neg[i] + ', '
            else:
                result += '<pad>' + ', '

        for i in range(EXAMPLE_NUM):
            if len(templete) > i:
                result += templete[i] + ', '
            else:
                result += '<pad>' + ', '

        result += str(regex) + '\n'


        max_len = max(max_len,len(raw))
        if data_num %1000 == 0:
            print('maxlen:', max_len)
            print(data_num)
        #print(idx)
        #print(result)
        # save_file.write(result)

        data_num += 1
        if opt.is_train:
            if data_num == 90000:
                break
        else:
            if data_num == 10000:
                break
    print('maxlen:', max_len)
def main():
    generate_data()


if __name__ == "__main__":
    main()
