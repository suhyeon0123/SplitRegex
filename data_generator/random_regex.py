from parsetree import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to save data', default='./data/random_regex')
parser.add_argument('--number', action='store', dest='number', type=int,
                    help='the number of data samples', default=10000)
parser.add_argument('--always_concat', action='store_true', dest='always_concat',
                    help='Indicate if the root of regex is always concat or not', default=False)

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

def main():
    get_regex_data(opt.number, opt.data_path)


if __name__ == "__main__":
    main()