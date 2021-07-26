from parsetree import*
from xeger import Xeger


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

        # regex의 leaf노드가 Concat이도록 함
        if regex.r.type != Type.C:
            continue
        if regex.starnormalform() or regex.redundant_concat1() or regex.redundant_concat2() or regex.KCK() or regex.KCQ() or regex.QC() or regex.OQ() or regex.orinclusive() or regex.prefix() or regex.sigmastar():
            continue
        print(regex)
        saved_regex = regex
        regex = regex.repr_labeled()
        print(regex)


        # pos examples 생성
        x = Xeger()
        posset = set()
        endcount = 0
        while endcount < 50 and len(posset) < 10:
            tmpexample = x.xeger(repr(regex))
            #print(tmpexample)
            if len(tmpexample) <= (10 + 2) and len(tmpexample) >= 3:
                posset.add(tmpexample)
            endcount += 1
        pos = [example.strip("'") for example in list(posset)]
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
                saved_decomposed_regex[-1] = saved_decomposed_regex[-1]+letter

        regex = "".join(str_list)
        save = regex





        # templetes 생성
        templete = []
        for example in pos:
            str_list = []
            print(regex)
            print(example)
            dic = re.fullmatch(regex, example).groupdict()
            print(dic)
            for i in range(1, len(dic)+1):
                key = "t" + str(i)
                targetstring = dic[key]
                if targetstring == None:
                    count = 0
                else:
                    count = len(targetstring)
                for _ in range(count):
                    str_list.append(str(i-1))
            templete.append("".join(str_list))
        print(templete)



        result = ''
        for i in range(10):
            if len(pos) > i:
                f.write(pos[i] + ', ')
                result = result + pos[i] + ', '
            else:
                f.write('<pad>' + ', ')
                result = result + '<pad>' + ', '

        for i in range(10):
            if len(templete) > i:
                f.write(templete[i] + ', ')
                result += templete[i] + ', '
            else:
                f.write('<pad>' + ', ')
                result += '<pad>' + ', '
        #f.write(str(bench_count))
        #f.write(''.join(list(map(lambda x: x+" ", list(str(bench_count))))))


        f.write(str(save) + '\n')
        result += str(save) + '\n'

        #f.write(str(saved_decomposed_regex) + '\n')
        #result += str(saved_decomposed_regex) + '\n'


        print(result)
        print(bench_count)
        bench_count += 1
        print(' ')

    #save in txt file


get_train_data(100000, "./data/train.csv")
get_train_data(20000, "./data/valid.csv")