from parsetree import *
from xeger import Xeger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', action='store', dest='data_type',
                    help='data type - pos_label or pos_neg', default='pos_label')
parser.add_argument('--regex_path', action='store', dest='regex_path',
                    help='Path to save data', default='./data/random_regex')
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to save data', default='./data/valid_5.csv')

opt = parser.parse_args()

limit = 6
alpha_size = 5


def rand_example(limit):
    regex = REGEX()
    for count in range(limit):
        regex.make_child(1)
    regex.spreadRand()
    return regex


# len(pos_example) <=10
def make_pos_label(regex_file, save_file):
    regex_file = open(regex_file, 'r')
    regexes = [x.strip() for x in regex_file.readlines()]

    save_file = open(save_file + '.csv', 'w')

    for regex in regexes:
        print(regex)
        # pos examples 생성
        x = Xeger()
        posset = set()

        for i in range(50):
            example = x.xeger(repr(regex)).strip("'")
            if 0 < len(example) <= 10:
                posset.add(example)
            if len(posset) == 10:
                break

        pos = list(posset)
        if len(pos) != 10:
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

        saved_decomposed_regex = []
        bracket = 0
        for letter in regex:
            if letter == '(':
                if bracket == 0:
                    saved_decomposed_regex.append('')
                else:
                    saved_decomposed_regex[-1] = saved_decomposed_regex[-1] + letter
                bracket += 1
            elif letter == ')':
                if bracket != 1:
                    saved_decomposed_regex[-1] = saved_decomposed_regex[-1] + letter
                bracket -= 1
            else:
                saved_decomposed_regex[-1] = saved_decomposed_regex[-1] + letter

        regex = "".join(str_list)
        save = regex

        # templetes 생성
        templete = []
        for example in pos:
            str_list = []
            dic = re.fullmatch(regex, example).groupdict()
            for i in range(1, len(dic) + 1):
                key = "t" + str(i)
                targetstring = dic[key]
                if targetstring == None:
                    count = 0
                else:
                    count = len(targetstring)
                for _ in range(count):
                    str_list.append(str(i - 1))
            templete.append("".join(str_list))

        # save as csv file
        result = ''
        for i in range(10):
            if len(pos) > i:
                result += pos[i] + ', '
            else:
                result += '<pad>' + ', '

        for i in range(10):
            if len(templete) > i:
                result += templete[i] + ', '
            else:
                result += '<pad>' + ', '

        result += str(save) + '\n'

        print(result)
        save_file.write(result)


def make_pos_neg(regex_file, save_file):
    regex_file = open(regex_file, 'r')
    regexes = [x.strip() for x in regex_file.readlines()]

    f = open(save_file + '.csv', 'w')

    for regex in regexes:

        # pos examples 생성
        x = Xeger()
        posset = set()

        for i in range(50):
            example = x.xeger(repr(regex)).strip("'")
            if 0 < len(example) <= 10:
                posset.add(example)
            if len(posset) == 10:
                break

        pos = list(posset)
        if len(pos) != 10:
            continue

        # neg examples 생성
        negset = set()
        for i in range(0, 1000):
            # random regex생성
            str_list = []

            for j in range(0, random.randrange(1, 10)):
                str_list.append(str(random.randrange(0, 5)))
            tmp = ''.join(str_list)

            # random regex가 맞지 않다면 추가
            if not bool(re.fullmatch(repr(regex), tmp)):
                negset.add(tmp)

            if len(negset) == 10:
                break

        if not len(negset) == 10:
            continue
        neg = list(negset)

        # save as csv file
        result = ''
        for i in range(10):
            if len(pos) > i:
                result += pos[i] + ', '
            else:
                result += '<pad>' + ', '

        for i in range(10):
            if len(neg) > i:
                result += neg[i] + ', '
            else:
                result += '<pad>' + ', '

        result += str(regex) + '\n'

        print(result)
        f.write(result)


def main():
    if opt.data_type == 'pos_label':
        make_pos_label(opt.regex_path, opt.data_path)
    else:
        make_pos_neg(opt.regex_path, opt.data_path)


if __name__ == "__main__":
    main()
