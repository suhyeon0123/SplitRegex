from parsetree import *
from xeger import Xeger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to save data', default='./data/pos_neg_5.csv')
parser.add_argument('--number', action='store', dest='number', type=int,
                    help='the number of data samples', default=1000)
opt = parser.parse_args()

limit = 6


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

        if regex.starnormalform() or regex.redundant_concat1() or regex.redundant_concat2() or regex.KCK() or regex.KCQ() or regex.QC() or regex.OQ() or regex.orinclusive() or regex.prefix() or regex.sigmastar():
            continue

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

        print(bench_count)
        print(result)
        f.write(result)
        bench_count += 1


def main():
    get_train_data(opt.number, opt.data_path)


if __name__ == "__main__":
    main()
