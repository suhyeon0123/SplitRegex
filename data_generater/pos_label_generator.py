from parsetree import *
from xeger import Xeger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to save data', default='./data/valid_5.csv')
parser.add_argument('--number', action='store', dest='number', type=int,
                    help='the number of data samples', default=10000)
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
def get_train_data(bench_num, file_name):
    f = open(file_name, 'w')

    bench_count = 0
    while bench_count < bench_num:

        # REGEX 생성
        regex = rand_example(limit)

        # regex의 leaf노드가 Concat이도록 함
        if regex.r.type != Type.C:
            continue

        if regex.starnormalform() or regex.redundant_concat1() or regex.redundant_concat2() or regex.KCK() or regex.KCQ() or regex.QC() or regex.OQ() or regex.orinclusive() or regex.prefix() or regex.sigmastar():
            continue

        regex = regex.repr_labeled()

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

        print(bench_count)
        print(result)
        f.write(result)
        bench_count += 1


def main():
    get_train_data(opt.number, opt.data_path)


if __name__ == "__main__":
    main()
