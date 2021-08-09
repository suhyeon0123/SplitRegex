import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodels', 'SoftConciseNormalForm')))
from parsetree import *
from xeger import Xeger
import argparse
from synthesizer import synthesis
from examples import Examples

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to save data', default='./data/random_regex')
parser.add_argument('--number', action='store', dest='number', type=int,
                    help='the number of data samples', default=10000)
parser.add_argument('--always_concat', action='store_true', dest='always_concat',
                    help='Indicate if the root of regex is always concat or not', default=False)
parser.add_argument('--alphabet_size', action='store', dest='alphabet_size',
                    help='define the alphabet size of the regex', default=5)
parser.add_argument('--is_compact', action='store_true', dest='is_compact',
                    help='use more compact regex', default=False)



opt = parser.parse_args()

limit = 6


def rand_example(limit):
    regex = REGEX()
    for count in range(limit):
        regex.make_child(1)
    regex.spreadRand()
    return regex


def get_regex_data(bench_num, file_name):
    f = open(file_name, 'w')

    bench_count = 0
    while bench_count < bench_num:

        # REGEX 생성
        regex = rand_example(limit)

        # regex의 leaf노드가 Concat이도록 함
        if opt.always_concat and regex.r.type != Type.C:
            continue

        if regex.starnormalform() or regex.redundant_concat1() or regex.redundant_concat2() or regex.KCK() or regex.KCQ() or regex.QC() or regex.OQ() or regex.orinclusive() or regex.prefix() or regex.sigmastar():
            continue

        print(str(bench_count) + ': ' + regex.repr_labeled())
        f.write(str(regex.repr_labeled()) + '\n')
        bench_count += 1

def get_compact_regex_data(bench_num, file_name):
    f = open(file_name, 'w')

    bench_count = 0
    while bench_count < bench_num:

        # REGEX 생성
        regex = rand_example(limit)

        # regex의 leaf노드가 Concat이도록 함
        if opt.always_concat and regex.r.type != Type.C:
            continue

        if regex.starnormalform() or regex.redundant_concat1() or regex.redundant_concat2() or regex.KCK() or regex.KCQ() or regex.QC() or regex.OQ() or regex.orinclusive() or regex.prefix() or regex.sigmastar():
            continue

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

        compact_regex = synthesis(Examples(pos=pos, neg=neg), 3000)

        if compact_regex is not None:
            print(str(bench_count) + ': ' + regex.repr_labeled() + ' -> '+compact_regex.repr_labeled())
            f.write(str(compact_regex.repr_labeled()) + '\n')
            bench_count += 1
        else:
            print(str(bench_count) + ': ' + regex.repr_labeled())
            f.write(str(regex.repr_labeled()) + '\n')
            bench_count += 1



def main():
    if opt.is_compact:
        get_compact_regex_data(opt.number, opt.data_path)
    else:
        get_regex_data(opt.number, opt.data_path)


if __name__ == "__main__":
    main()