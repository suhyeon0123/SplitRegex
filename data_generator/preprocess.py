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
        example = x.xeger(regex).strip("'")
        posset.add(example)
    pos = list(posset)

    for i in range(number - len(pos)):
        pos.append('<pad>')
    return pos


def make_neg(regex, number):
    negset = set()
    for i in range(0, 1000):
        # random regex생성
        str_list = []

        for j in range(0, random.randrange(5, 10)):
            str_list.append(chr(random.randrange(33, 91)))
        tmp = ''.join(str_list)


        # random regex가 맞지 않다면 추가
        if re.fullmatch(regex, tmp) is None:
            negset.add(tmp)

        if len(negset) == 10:
            break

    return list(negset)


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



def preprocess_control_ascii(regex):
    return re.sub(r'\\x([01][0-9A-Fa-f])', r'(`\1)', regex)


def preprocess_space_character(regex):
    regex = re.sub(r'\\r', r'(`20)', regex)
    regex = re.sub(r'\\n', r'(`21)', regex)
    regex = re.sub(r'\\t', r'(`22)', regex)

    regex = re.sub(r'\\x5(c|C)', r'(`5c)', regex)

    return regex


def preprocess_character_set(regex):
    #regex = re.sub(r'\\w', r'(`30)', regex)
    regex = re.sub(r'\\W', r'(`31)', regex)
    #regex = re.sub(r'\\d', r'(`32)', regex)
    regex = re.sub(r'\\D', r'(`33)', regex)
    regex = re.sub(r'\\s', r'(`34)', regex)
    #regex = re.sub(r'\\S', r'(`35)', regex)

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
                ch = chr(idx + 97)
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

        # if idx!=1180:
        #     continue

        print(regex)


        regex = remove_anchor(regex)
        regex = remove_redundant_quantifier(regex)
        regex = preprocess_parenthesis_flag(regex)
        print(regex)

        regex = get_captured_regex(regex)
        print(regex)

        #make vocab
        regex = preprocess_control_ascii(regex)
        regex = preprocess_space_character(regex)
        regex = preprocess_character_set(regex)
        print(regex)

        regex = replace_constant_string(regex)
        regex = re.sub('`1`', '\\\(', regex)
        regex = re.sub('`2`', '\\\)', regex)
        print(regex)


        try:
            pos = make_pos(regex, 10)
            print(pos)
        except:
            error_idx.append(idx)
            continue

        neg = make_neg(regex, 10)
        print(neg)

        label = make_label(regex, pos)
        print(label)

        total = pos + neg + label
        total.append(regex)
        print(total)
        res = ''
        for ele in total:
            res = res + str(ele) + '\t'

        print(res)
        save_file.write(res)
        print('')

    print(error_idx)


if __name__ == '__main__':
    main()