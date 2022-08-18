import os, sys
from random import random
import re2 as re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'submodels', 'SoftConciseNormalForm')))
from parsetree import *
from xeger import Xeger
import argparse
import configparser
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--alphabet_size', action='store', dest='alphabet_size',
                    help='define the alphabet size of the regex', type=int, default=5)
parser.add_argument('--is_train', action='store_true', dest='is_train',
                    help='Indicate if the data is used for train or test', default=False)
parser.add_argument('--number', action='store', dest='number', type=int,
                    help='the number of data samples', default=10000)

opt = parser.parse_args()


MAX_SEQUENCE_LENGTH = 10
EXAMPLE_NUM = 20
TRAIN_SEED = 1515
TEST_SEED = 1616
max_depth = 4


def generate_rand_regex(alphabet_size=5):
    regex = REGEX()
    for _ in range(max_depth):
        regex.make_child(alphabet_size=alphabet_size)
    regex.spreadRand(alphabet_size=alphabet_size)
    return regex


# generate random regex
def get_concise_regex():
    while True:
        regex = generate_rand_regex(opt.alphabet_size)

        if regex.starnormalform() or regex.redundant_concat1() or regex.redundant_concat2(
                opt.alphabet_size) or regex.KCK(opt.alphabet_size) or regex.KCQ(
            opt.alphabet_size) or regex.QC() or regex.OQ() or regex.orinclusive(
            opt.alphabet_size) or regex.prefix() or regex.sigmastar(opt.alphabet_size):
            continue
        else:
            break

    return regex.repr_labeled()


def get_pos(regex, xeger):
    pos = []

    for i in range(200):
        example = xeger.xeger(regex).strip("'")
        if 0 < len(example) <= MAX_SEQUENCE_LENGTH and example not in pos:
            pos.append(example)
        if len(pos) == EXAMPLE_NUM:
            break

    return pos


def get_neg(regex):
    neg = []
    for _ in range(1000):

        # generate random strings
        str_list = []
        for j in range(random.randrange(1, MAX_SEQUENCE_LENGTH + 1)):
            str_list.append(str(random.randrange(0, opt.alphabet_size)))
        tmp = ''.join(str_list)

        # add random string if it deny regex
        if not bool(re.fullmatch(regex, tmp)) and tmp not in neg:
            neg.append(tmp)

        if len(neg) == EXAMPLE_NUM:
            break
    return neg





def pt_add(regex):
    if 't2' in regex and 't9' not in regex:
        while True:
            new_subregex = generate_rand_regex(10, 2)
            if new_subregex.r.type == Type.C:
                continue
            else:
                break

        start, end = random.sample([(m.start(0), m.end(0)) for m in re.finditer('\(\?P\<.*?\)(?=\()', regex)], 1)[0]    
        tmp = regex[start:]
        idx = re.findall('\<t\d', tmp)[0]
        for point in reversed(re.findall('\<t\d', tmp)):
            tmp = re.sub(point, '<t' + str(int(point[2])+1), tmp)
        regex = regex[:start] + '(?P<t' + idx[2]  + '>'+  repr(new_subregex)  + ')' + tmp

    return regex

def pt_remove(regex):

    if 't2' in regex and 't9' not in regex:
        start, end = random.sample([(m.start(0), m.end(0)) for m in re.finditer('\(\?P\<.*?\)(?=\()', regex)], 1)[0]    
        tmp = regex[end:]
        for point in re.findall('\<t\d', tmp):
            tmp = re.sub(point, '<t' + str(int(point[2])-1), tmp)
        regex = regex[:start] + tmp
        
    return regex

def pt_substitute(regex):
    index = random.sample([m.start(0) for m in re.finditer('(?<!t)\d', regex)], 1)[0]
    regex = regex[:index] + str(random.randint(0,9)) + regex[index + 1:]
    return regex

def perturb_regex(regex):

    func_list = [pt_add, pt_remove, pt_substitute]
    
    for _ in range(random.randint(1,3)):
        regex = func_list[random.randint(0,2)](regex)


    return regex


def get_neg2(regex, xeger):
    neg = []
    for _ in range(1000):
        
        perturbed_regex = perturb_regex(regex)
        example = xeger.xeger(perturbed_regex).strip("'")
        if 0 < len(example) <= MAX_SEQUENCE_LENGTH and example not in neg and not bool(re.fullmatch(regex, example)) and example not in neg:
            neg.append(example)

        if len(neg) == EXAMPLE_NUM:
            break
    return neg


def attach_tag(regex):
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
    return "".join(str_list)


def split_regex(regex):
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
    return subregex_list


def labeling(regex, pos, subregex_list):
    SIGMA_STAR = '0'

    templete = []
    for example in pos:
        str_list = []
        dic = re.fullmatch(regex, example).groupdict()
        label_num = 1
        for i in range(1, len(dic) + 1):

            targetstring = dic['t' + str(i)]
            targetregex = re.sub('\?P\<t\d*?\>', '', subregex_list[i - 1])

            if targetregex == str(KleenStar(Or(*[Character(str(x)) for x in range(opt.alphabet_size)]))):
                label = SIGMA_STAR
            else:
                if label_num < 10:
                    label = str(label_num)
                else:
                    label = chr(55+label_num)
            label_num += 1

            count = len(targetstring)

            for _ in range(count):
                str_list.append(label)
        templete.append("".join(str_list))
    return templete


def generate_data():

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    random.seed(int(config['seed']['random_data']) + int(opt.alphabet_size))
    xeger = Xeger(limit=5)
    xeger.seed(int(config['seed']['random_data']) + int(opt.alphabet_size))


    pathlib.Path('./data/regex_perturb/random_data/size_' + str(opt.alphabet_size)).mkdir(parents=True, exist_ok=True)
    if opt.is_train:
        save_file = open('./data/regex_perturb/random_data/size_' + str(opt.alphabet_size) + '/train.csv', 'w+')
        random.seed(int(config['seed']['random_data']) + int(opt.alphabet_size) + TRAIN_SEED+1) 
        xeger.seed(int(config['seed']['random_data']) + int(opt.alphabet_size) + TRAIN_SEED+1)
    else:
        save_file = open('./data/regex_perturb/random_data/size_' + str(opt.alphabet_size) + '/test.csv', 'w+')
        random.seed(int(config['seed']['random_data']) + int(opt.alphabet_size) + TEST_SEED+1)
        xeger.seed(int(config['seed']['random_data']) + int(opt.alphabet_size) + TEST_SEED+1)

    data_num = 0
    max_len = 0
    while data_num < opt.number:

        # generate random regex
        regex = get_concise_regex()
        raw = regex

        # generate pos examples
        pos = get_pos(regex, xeger)
        if len(pos) != EXAMPLE_NUM:
            continue

        # generate neg examples
        neg = get_neg2(regex, xeger)
        if not len(neg) == EXAMPLE_NUM:
            continue

        # Tag preprocess
        regex = attach_tag(regex)
        subregex_list = split_regex(regex)


        # make label from the pos strings
        templete = labeling(regex, pos, subregex_list)


        # save as csv file
        result = ''
        result += ', '.join(pos) + ', '
        result += ', '.join(neg) + ', '
        result += ', '.join(templete) + ', '
        result += str(regex) + '\n'

        # print(str(data_num) + ': ' + result)
        save_file.write(result)
        data_num += 1
        max_len = max(max_len, len(raw))


    print('maxlen:', max_len)


def main():
    generate_data()


if __name__ == "__main__":
    main()
