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

dequantifier = '(\[[^]]*\]|\\\.|\\.|\\\\x..)'
quantifier = '(\*|\+|\?|\{\d+\,\d*\}|\{\d+\})\??'

# parenthesis
dequantifier2 = '\(.*?\)'

dequantifier3 = '(\([^(\))]*?\([^(\))]*?|[^(\()]*?\)[^(\()]*?\)|\(.+?\|.+?\))'

dequantifier4 = '\[[^]]*\]'
dequantifier5 = '\\\\d|\\\\D\\\\|\\\\w|\\\\W|\\\\s|\\\\S|(?<!\\\\)\.'


def make_pos(regex, number):
    x = Xeger()
    posset = set()

    for i in range(number):
        posset.add(x.xeger(regex))
    pos = list(posset)

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
                    str_list.append(str(i))
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
    regex = re.sub('/\^', '/', regex)
    regex = re.sub('=\^', '=', regex)

    regex = re.sub('\$/', '/', regex)
    regex = re.sub('php\$', 'php', regex)

    regex = re.sub('\(\^\|(\&)\)', r'\1', regex)
    regex = re.sub('\(([^\(]*?)\|\$\)', r'\1', regex)
    regex = re.sub('\(\$\|(.*?)\)', r'\1', regex)

    return regex


def remove_redundant_quantifier(regex):
    regex = re.sub('}\+', '}', regex)

    while True:
        regex, a = re.subn('(\[[^]]*\]|\\\.|\\.|\(.*?\))' + '((\+|\{\d+\,\d*\}|\{\d+\})\??|\*\?)', r'\1*', regex)
        regex, b = re.subn('(\[[^]]*\]|\\\.|\\.|\(.*?\))' + '(\?\?)', r'\1?', regex)
        regex, c = re.subn(r'(\\x[0-9A-Fa-f][0-9A-Fa-f]|@)' + '((\+|\?|\{\d+\,\d*\}|\{\d+\})\??|\*\?)', r'\1*', regex)

        if a + b + c == 0:
            break

    regex = re.sub(r'\\b', r'', regex)
    regex = re.sub(r'\\B', r'', regex)

    # remove back reference
    regex = re.sub(r'\\\d', r'', regex)

    return regex


def preprocess_parenthesis_flag(regex):
    regex = re.sub(r'\(\?:', '(', regex)
    regex = re.sub(r'\(\?=', '(', regex)
    regex = re.sub(r'\(\?!', '(', regex)
    regex = re.sub(r'\(\?<.*?>', '(', regex)
    regex = re.sub(r'\\k<.*?>', '', regex)

    regex = re.sub(r'\\\(', r'`1`', regex)
    regex = re.sub(r'\\\)', r'`2`', regex)

    return regex



def preprocess_replace(regex):
    # control_ascii
    regex = re.sub(r'\\x([01][0-9A-Fa-f])', r'(`\1)', regex)

    # space_character
    regex = re.sub(r'\\r', r'(`20)', regex)
    regex = re.sub(r'\\n', r'(`21)', regex)
    regex = re.sub(r'\\t', r'(`22)', regex)

    regex = re.sub(r'\\x5(c|C)', r'(`5c)', regex)

    # character_set
    regex = re.sub(r'\\w', r'(`30)', regex)
    # regex = re.sub(r'\\W', r'(`31)', regex)
    regex = re.sub(r'\\d', r'(`32)', regex)
    # regex = re.sub(r'\\D', r'(`33)', regex)
    # regex = re.sub(r'\\s', r'(`34)', regex)
    # regex = re.sub(r'\\S', r'(`35)', regex)

    return regex


def get_captured_regex(regex):
    matchObj_iter = re.finditer(dequantifier + quantifier + '|' + dequantifier2 + quantifier + '|' + dequantifier3 + '|' +dequantifier4 + '|' + dequantifier5, regex)

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

    for idx, subregex in enumerate(subregex_list):
        if re.search(
                dequantifier + quantifier + '|' + dequantifier2 + quantifier + '|' + dequantifier3 + '|' + dequantifier4 + '|' + dequantifier5 + '|' + '\(`3\d\)',
                subregex) is None:

            if idx < 26:
                ch = chr(idx + 65)
            else:
                ch = chr(idx + 71)

            regex = re.sub(subregex, ch, regex, 1)
            subregex_list[idx] = ch

        if re.fullmatch('\(.*\)', subregex_list[idx]) is None:
            subregex_list[idx] = '(' + subregex_list[idx] + ')'

    return ''.join(subregex_list)



def main():
    if opt.data_type == 'snort':
        regex_file = open('../submodels/automatark/regex/snort-clean.re', 'r')
    else:
        regex_file = open('snort-clean.re', 'r')

    save_file = open(opt.data_path, 'w')
    regex_list = [x.strip() for x in regex_file.readlines()]
    error_idx = []

    for idx, regex in enumerate(regex_list):
        if idx==902 or idx==903:
            continue
        print(regex)

        regex = remove_anchor(regex)
        regex = remove_redundant_quantifier(regex)
        regex = preprocess_parenthesis_flag(regex)

        regex = get_captured_regex(regex)

        regex = replace_constant_string(regex)
        regex = re.sub('`1`', '\\\(', regex)
        regex = re.sub('`2`', '\\\)', regex)
        print(regex)


        # generate pos strings
        try:
            pos = make_pos(regex, 10)
        except:
            error_idx.append(idx)
            continue

        # generate neg strings
        neg = make_neg(regex, pos, 10)

        # generate label
        try:
            label = make_label(regex, pos)
        except:
            error_idx.append(idx)
            continue

        # replace unrecognized symbol
        pos = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), pos))
        neg = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), neg))

        print(pos)
        print(neg)
        print(label)
        print('')

        total = pos + neg + label
        total.append(regex)

        res = ''
        for ele in total:
            res = res + str(ele) + '\t'

        save_file.write(res+'\n')


    print(error_idx)


if __name__ == '__main__':
    main()