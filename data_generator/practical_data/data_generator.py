from xeger import Xeger

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'submodels', 'SoftConciseNormalForm')))

import re2 as re
import random

dequantifier = '(\\\.|\\.|\\\\x..)'
dequantifier2 = '(\(.*?\)|\[[^]]*\])'

quantifier = '(\*|\+|\?|\{\d+\,\d*\}|\{\d+\})\??'

dequantifier5 = '\\\\d|\\\\D\\\\|\\\\w|\\\\W|\\\\s|\\\\S|(?<!\\\\)\.'

special_quantifier = '!\*'

def make_pos(regex, number):
    x = Xeger()
    posset = set()

    for i in range(number):
        tmp = x.xeger(regex)
        if len(tmp) < 30:
            posset.add(tmp)

    pos = list(filter(None, list(posset)))

    for i in range(number - len(pos)):
        pos.append('<pad>')

    return pos


def make_label(regex, pos):
    # Tag 전처리
    str_list = []
    bracket = 0
    tagIndex = 0
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

    # templetes 생성
    templete = []

    for example in pos:
        if example != '<pad>':
            str_list = []

            example = re.sub('\x0b', '`', example)

            xx = re.fullmatch(regex, example)
            if xx is None:

                example = re.sub('`', '\t', example)

            dic = re.fullmatch(regex, example).groupdict()

            for i in range(len(dic)):
                key = "t" + str(i)
                targetstring = dic[key]
                if targetstring == None:
                    count = 0
                else:
                    count = len(targetstring)
                for _ in range(count):
                    if i < 10:
                        label = str(i)
                    else:
                        label = chr(55+i)
                    str_list.append(label)
            templete.append("".join(str_list))
        else:
            templete.append('<pad>')

    return templete


def make_neg(regex, pos, number):
    negset = set()

    symbol_list = set()
    for i in pos:
        if i == '<pad>':
            continue
        x = re.findall('(?!\x0b|\\\\|\\|\').', i)
        for j in x:
            symbol_list.add(j)

    symbol_list = list(symbol_list)

    for i in range(0, 1000):
        # select random pos
        example = pos[random.randrange(0, len(pos))]
        if example == '<pad>':
            continue

        if len(example) <= 10:
            count = 2
        else:
            count = int(len(example)/5)

        for j in range(count):
            point = random.randrange(0, len(example))
            if example[point] != "'" and example[point] != r"\\":
                example = example[:point] + symbol_list[random.randrange(0, len(symbol_list))] + example[point+1:]


        # random regex가 맞지 않다면 추가
        if re.fullmatch(regex, example) is None:
            negset.add(example)

        if len(negset) == 10:
            break

    neg = list(negset)
    for i in range(number - len(negset)):
        neg.append('<pad>')

    return neg




def remove_anchor(regex):
    regex = re.sub(r'(?<!\x5c|\[)\^', '', regex)
    regex = re.sub(r'(?<!\x5c)\$', '', regex)

    return regex


def remove_redundant_quantifier(regex):
    regex = re.sub('}\+', '}', regex)

    while True:
        regex, a = re.subn('(\[[^]]*\]|\\\.|\\.|\(.*?\))' + '((\+|\{\d+\,\d*\}|\{\d+\})\??|\*\?)', r'\1*', regex)
        regex, b = re.subn('(\[[^]]*\]|\\\.|\\.|\(.*?\))' + '(\?\?)', r'\1?', regex)
        regex, c = re.subn(r'(\\x[0-9A-Fa-f][0-9A-Fa-f]|@)' + '((\+|\?|\{\d+\,\d*\}|\{\d+\})\??|\*\?)', r'\1*', regex)
        if a + b + c == 0:
            break

    # remove back reference
    regex = re.sub(r'\\\d', r'', regex)

    return regex


def preprocess_parenthesis_flag(regex):
    regex = re.sub(r'\(\?:', '(', regex)
    regex = re.sub(r'\(\?=', '(', regex)
    regex = re.sub(r'\(\?<=', '(', regex)
    regex = re.sub(r'\(\?!', '(', regex)
    regex = re.sub(r'\(\?<!', '(', regex)
    regex = re.sub(r'\(\?<.*?>', '(', regex)
    regex = re.sub(r'\(\?P<.*?>', '(', regex)

    regex = re.sub(r'\\b', r'', regex)
    regex = re.sub(r'\\B', r'', regex)
    regex = re.sub(r'\\k', '', regex)


    regex = re.sub(r'\\\[', r'!', regex)
    regex = re.sub(r'\\\]', r'!', regex)
    regex = re.sub(r'\\\(', r'!', regex)
    regex = re.sub(r'\\\)', r'!', regex)

    # remove not operator
    regex = re.sub('(?<=\[)\^([^]]*?)(?=\])', r'\1', regex)
    regex = re.sub('\\\s', '`', regex)

    return regex



def preprocess_replace(regex):
    # control_ascii
    regex = re.sub(r'\\x([01][0-9A-Fa-f])', r'!', regex)

    # space_character
    regex = re.sub(r'\\r', r'!', regex)
    regex = re.sub(r'\\n', r'!', regex)
    regex = re.sub(r'\\t', r'!', regex)
    regex = re.sub(r' ', r'!', regex)
    regex = re.sub(r'#', r'!', regex)
    regex = re.sub(r',', r'!', regex)
    regex = re.sub(r'\\\\', r'!', regex)
    regex = re.sub(r'\\\'', r'!', regex)
    regex = re.sub(r'\\', r'!', regex)

    regex = re.sub(r'\\x5(c|C)', r'!', regex)

    regex = re.sub('(?<!pad)[^0-9a-zA-Z](?!pad)', r'!', regex)

    return regex


def get_captured_regex(regex):
    matchObj_iter = re.finditer(dequantifier + quantifier + '|' + dequantifier2 + '(' + quantifier + ')?' + '|' + dequantifier5 , regex)

    split_point = [0]
    indicate = 1

    regex = '(' + regex
    for matchObj in matchObj_iter:
        start, end = matchObj.start(), matchObj.end()
        regex = regex[:start + indicate] + ')' + regex[start + indicate:]
        indicate += 1
        regex = regex[:start + indicate] + '(' + regex[start + indicate:]
        indicate += 1
        regex = regex[:end + indicate] + ')' + regex[end + indicate:]
        indicate += 1
        regex = regex[:end + indicate] + '(' + regex[end + indicate:]
        indicate += 1
        split_point.append(start)
        split_point.append(end)
    regex = regex + ')'

    regex = re.sub('\(\)', '', regex)

    return regex


def replace_constant_string(regex):
    mapping_table = {}

    # make subregex list
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

    # replace 1st strings
    for idx, subregex in enumerate(subregex_list):
        if re.search(
                dequantifier + quantifier + '|' + dequantifier2 + '(' + quantifier + ')?' + '|' + dequantifier5 ,
                subregex) is None:
            if subregex in mapping_table.values():
                for alphabet, string in mapping_table.items():
                    if string == subregex:
                        ch = alphabet
            else:
                if len(mapping_table) < 26:
                    ch = chr(len(mapping_table) + 65)
                else:
                    ch = chr(len(mapping_table) + 71)
                mapping_table[ch] = subregex
            regex = re.sub(repr(subregex), ch, regex, 1)
            subregex_list[idx] = ch

        if re.fullmatch('\(.*\)', subregex_list[idx]) is None:
            subregex_list[idx] = '(' + subregex_list[idx] + ')'

    regex = ''.join(subregex_list)

    regex = re.sub('\\\\x..', '!', regex)
    regex = re.sub(r',', r'!', regex)

    string_pattern = '(?<!\\\\)[^\\\(\)\*\+\|\^\[\]\!\?]{2,}'
    while re.search(string_pattern, regex) is not None:
        tmp = re.search(string_pattern, regex).group()
        if tmp in mapping_table.values():
            for alphabet, string in mapping_table.items():
                if string == tmp:
                    ch = alphabet
        else:
            if len(mapping_table) < 26:
                ch = chr(len(mapping_table) + 65)
            else:
                ch = chr(len(mapping_table) + 71)
            mapping_table[ch] = tmp

        regex = re.sub(string_pattern, ch, regex, 1)

    return regex, mapping_table



def main():
    data_pathes = ['submodels/automatark/regex/snort-clean.re', 'submodels/automatark/regex/regexlib-clean.re', 'practical_data/practical_regexes.json']
    train_file = open('data/practical_data/train.csv', 'w')
    test_snort_file = open('data/practical_data/test_snort.csv', 'w')
    test_regexlib_file = open('data/practical_data/test_regexlib.csv', 'w')
    test_Davis_file = open('data/practical_data/test_practicalregex.csv', 'w')


    for data_idx, data_path in enumerate(data_pathes):

        regex_file = open(data_path, 'r')
        data_name = re.search('[^/]*?(?=\.r|\.j)', data_path).group()
        print('Preprocessing ' + data_name + '...')

        regex_list = [x.strip() for x in regex_file.readlines()]
        random.shuffle(regex_list)

        single_regex_count = 0
        error_idx = []

        for idx, regex in enumerate(regex_list):

            if data_name =='regexlib-clean':
                regex = re.sub(r'\\\\', '\x5c', regex)
            if data_name =='practical_regexes':
                regex = regex[1:-1]
                regex = re.sub(r'\\\\\\\\', '!', regex)
                regex = re.sub(r'\\\\', '\x5c', regex)
                regex = re.sub(r'\x00', '', regex)


            # preprocess
            try:
                regex = remove_anchor(regex)
                regex = remove_redundant_quantifier(regex)
                regex = preprocess_parenthesis_flag(regex)

                regex = get_captured_regex(regex)

                regex, mapping_table = replace_constant_string(regex)

            except:
                error_idx.append(idx)
                continue

            if regex == '(A)':
                single_regex_count += 1
                continue

            # generate pos, neg, label
            try:
                pos = make_pos(regex, 10)
                if not pos:
                    continue
                neg = make_neg(regex, pos, 10)

                label = make_label(regex, pos)
            except:
                error_idx.append(idx)
                continue

            # replace unrecognized symbol
            pos = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), pos))
            neg = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), neg))

            total = pos + neg + label

            res = ''
            for ele in total:
                res = res + str(ele) + ', '
            res = res + str(regex)


            # make train & test dataset
            if idx < int(len(regex_list)*0.9):
                train_file.write(res+'\n')
            else:
                if data_idx == 0:
                    test_snort_file.write(res+'\n')
                elif data_idx == 1:
                    test_regexlib_file.write(res + '\n')
                else:
                    test_Davis_file.write(res + '\n')

            print(idx)
        print(len(error_idx))
        print(single_regex_count)


if __name__ == '__main__':
    main()