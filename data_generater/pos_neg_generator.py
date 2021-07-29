from parsetree import*
from xeger import Xeger
import argparse

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
        endcount = 0
        while endcount < 50 and len(posset) < 10:
            tmpexample = x.xeger(repr(regex))
            if len(tmpexample) <= (10 + 2) and len(tmpexample) >= 3:
                posset.add(tmpexample)
            endcount += 1
        pos = [example.strip("'") for example in list(posset)]
        if len(pos) != 10:
            continue


        # neg examples 생성
        negset = set()
        for i in range(0, 1000):
            # random regex생성
            str_list = []

            for j in range(0, random.randrange(1, 15)):
                if random.random() < 0.5:
                    str_list.append('0')
                else:
                    str_list.append('1')
            tmp = ''.join(str_list)

            # random regex가 맞지 않다면 추가
            if not bool(re.fullmatch(repr(regex), tmp)):
                negset.add(tmp)

            if len(negset) == 10:
                break

        if not len(negset) == 10:
            continue
        neg = list(negset)



        result = ''
        for i in range(10):
            if len(pos) > i:
                f.write(pos[i] + ', ')
                result = result + pos[i] + ', '
            else:
                f.write('<pad>' + ', ')
                result = result + '<pad>' + ', '

        for i in range(10):
            if len(neg) > i:
                f.write(neg[i] + ', ')
                result += neg[i] + ', '
            else:
                f.write('<pad>' + ', ')
                result += '<pad>' + ', '


        f.write(str(regex) + '\n')
        result += str(regex) +'\n'

        print(result)
        print(bench_count)
        bench_count += 1
        print(' ')

    #save in txt file

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action='store', dest='data_path',
                    help='Path to save data', default='../data/pos_neg.csv')
parser.add_argument('--number', action='store', dest='number',
                    help='the number of data samples', default=10000)
opt = parser.parse_args()

get_train_data(opt.number, opt.data_path)
