from parsetree_makesplit import*
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

    bench_count = 1
    while bench_count <= bench_num:

        # REGEX 생성
        regex = rand_example(limit)
        # Concat의 학습비중을 높이기 위함
        if regex.r.type != Type.C and random.random() < 0.7:
            continue
        if regex.starnormalform() or regex.redundant_concat1() or regex.redundant_concat2() or regex.KCK() or regex.KCQ() or regex.QC() or regex.OQ() or regex.orinclusive() or regex.prefix() or regex.sigmastar():
            continue
        print(regex)
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
        if len(pos) == 0:
            continue
        print(pos)
        #print(pos[0])


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

        regex = "".join(str_list)
        #print(regex)



        # templetes 생성
        templete = []
        for example in pos:
            str_list = []
            dic = re.fullmatch(regex, example).groupdict()
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



        bench_count += 1
        result = ''

        for i in range(10):
            if len(pos) > i:
                f.write(pos[i].replace(""," ")[1:-1] + '\t')
                result = result + pos[i].replace(""," ")[1:-1] + '\t'
            else:
                f.write('<pad>' + '\t')
                result = result + '<pad>' + '\t'

        for i in range(10):
            if len(templete) > i:
                f.write(templete[i].replace("", " ")[1:-1] + '\t')
                result += templete[i].replace("", " ")[1:-1] + '\t'
            else:
                f.write('<pad>' + '\t')
                result += '<pad>' + '\t'
        f.write('\n')

        print(result)
        print(bench_count)
        print(' ')

    #save in txt file



get_train_data(100000, "/home/ksh/PycharmProjects/train4.txt")
get_train_data(20000, "/home/ksh/PycharmProjects/valid4.txt")

