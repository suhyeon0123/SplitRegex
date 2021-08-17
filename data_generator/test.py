from xeger import Xeger
import re2 as re
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', action='store', dest='data_type',
                    help='data type - snort or regexlib', default='snort')
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to save data', default='../data/snort.csv')
opt = parser.parse_args()

dequantifier = '(\[[^]]*\]|\\\.)'
quantifier = '(\*|\+|\?|\{\d+\,\d*\}|\{\d+\})\??'


def get_captured_regex(regex):
    # remove anchor
    regex = re.sub('/\^', '/', regex)
    regex = re.sub('=\^', '=', regex)

    regex = re.sub('\$/', '', regex)
    regex = re.sub('\$|', '', regex)
    regex = re.sub('|\$', '', regex)

    # remove \r, \s
    # regex = re.sub('\\x0d', r'\\t', regex)
    # regex = re.sub('\\\r', r'\\t', regex)
    # regex = re.sub('\\\s', r'\\t', regex)

    matchObj_iter = re.finditer(dequantifier + quantifier, regex)

    split_point = [0]
    indicate = 1

    regex = '(' + regex
    for matchObj in matchObj_iter:
        regex = regex[:matchObj.start() + indicate] + ')' + regex[matchObj.start() + indicate:]
        indicate += 1
        regex = regex[:matchObj.start() + indicate] + '(' + regex[matchObj.start() + indicate:]
        indicate += 1
        regex = regex[:matchObj.end() + indicate] + ')' + regex[matchObj.end() + indicate:]
        indicate += 1
        regex = regex[:matchObj.end() + indicate] + '(' + regex[matchObj.end() + indicate:]
        indicate += 1
        split_point.append(matchObj.start())
        split_point.append(matchObj.end())
    regex = regex + ')'

    return regex

def make_pos(regex, number):
    x = Xeger()
    posset = set()
    for i in range(number):
        example = x.xeger(regex).strip("'")
        posset.add(example)
    return list(posset)


def make_neg(regex, number):
    negset = set()
    for i in range(0, 1000):
        # random regex생성
        str_list = []

        for j in range(0, random.randrange(1, 10)):
            str_list.append(str(random.randrange(0, 10)))
        tmp = ''.join(str_list)

        # random regex가 맞지 않다면 추가
        if not bool(re.fullmatch(regex, tmp)):
            negset.add(tmp)

        if len(negset) == 10:
            break

    neg = list(negset)


def make_label(regex, pos):
    # templetes 생성
    debug = []
    templete = []
    for example in pos:
        labeled_string = []
        groups = re.fullmatch(regex, example).groups()
        debug.append(groups)

        for idx, substring in enumerate(groups):
            for count in range(len(substring)):
                labeled_string.append(str(idx))

        templete.append("".join(labeled_string))
    print(debug)
    print(templete)



def main():
    if opt.data_type == 'snort':
        regex_file = open('../submodels/automatark/regex/snort-clean.re', 'r')
    else:
        regex_file = open('snort-clean.re', 'r')

    regex_list = [x.strip() for x in regex_file.readlines()]
    save_file = open(opt.data_path, 'w')

    for idx, regex in enumerate(regex_list):
        print(regex)
        regex = get_captured_regex(regex)
        print(regex)

        # generate pos strings
        pos = make_pos(regex, 20)
        # generate pos strings
        neg = make_neg(regex, 20)
        # generate label from pos strings
        label = make_label(regex, pos)

        print('')
        #save_file.write()

if __name__ == '__main__':
    main()