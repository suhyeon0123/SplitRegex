import os, sys
import re2 as re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'submodels', 'SoftConciseNormalForm')))
from parsetree import *
from xeger import Xeger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alphabet_size', action='store', dest='alphabet_size',
                    help='define the alphabet size of the regex', type=int, default=5)
parser.add_argument('--is_train', action='store_true', dest='is_train',
                    help='Indicate if the data is used for train or test', default=False)

opt = parser.parse_args()


MAX_SEQUENCE_LENGTH = 15
EXAMPLE_NUM = 20


def generate_data():

    if opt.is_train:
        DATA_USAGE = 'train'
    else:
        DATA_USAGE = 'test'

    regex_file = open('./data/random_data/size_' + str(opt.alphabet_size) + '/' + DATA_USAGE + '_regex', 'r')
    save_file = open('./data/random_data/size_' + str(opt.alphabet_size) + '/' + DATA_USAGE + '.csv', 'w')
    regexes = [x.strip() for x in regex_file.readlines()]

    for idx, regex in enumerate(regexes):

        # pos examples 생성
        x = Xeger()
        posset = set()

        for i in range(200):
            example = x.xeger(regex).strip("'")
            if 0 < len(example) <= MAX_SEQUENCE_LENGTH:
                posset.add(example)
            if len(posset) == EXAMPLE_NUM:
                break

        pos = list(posset)
        if len(pos) != EXAMPLE_NUM:
            continue


        # neg examples 생성
        negset = set()
        for _ in range(1000):

            # random regex생성
            str_list = []
            for j in range(random.randrange(1, EXAMPLE_NUM+1)):
                str_list.append(str(random.randrange(0, opt.alphabet_size)))
            tmp = ''.join(str_list)

            # random regex가 맞지 않다면 추가
            if not bool(re.fullmatch(regex, tmp)):
                negset.add(tmp)

            if len(negset) == EXAMPLE_NUM:
                break

        if not len(negset) == EXAMPLE_NUM:
            continue
        neg = list(negset)


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

        print(idx)
        print(result)
        save_file.write(result)



def main():
    generate_data()


if __name__ == "__main__":
    main()
