from queue import PriorityQueue
from util import *
import argparse
from examples import*
import time
import copy
import sys
import faulthandler

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--examples", type=int,
                    help="Example number")
parser.add_argument("-r", "--redundant", type=bool)
parser.add_argument("-n", "--normalform", default='none')
args = parser.parse_args()


sys.setrecursionlimit(5000000)
faulthandler.enable()



w = PriorityQueue()
scanned = set()
w.put((REGEX().getCost(), REGEX()))

if args.examples:
    examples = Examples(args.examples)
else:
    examples = Examples('AlphaRegex',5)
answer = examples.getAnswer()

print(examples.getPos(), examples.getNeg())

i = 0
traversed = 1
start = time.time()
prevCost = 0

finished = False


while not w.empty() and not finished:
    tmp = w.get()
    s = tmp[1]
    cost = tmp[0]

    prevCost = cost
    hasHole = s.hasHole()

    print("state : ", s, " cost: ",cost)
    if hasHole:
        for j, new_elem in enumerate([Character('0'), Character('1'), Or(),  Or(Character('0'),Character('1')), Concatenate(Hole(),Hole()), KleenStar(), Question()]):

            #print(repr(s), repr(new_elem))

            k = copy.deepcopy(s)

            if not k.spread(new_elem):
                #print("false "+ new_elem)
                continue

            traversed += 1
            if repr(k) in scanned:
                # print("Already scanned?", repr(k))
                # print(list(scanned))
                continue
            else:
                scanned.add(repr(k))

            if not k.hasHole():
                if is_solution(repr(k), examples, membership):
                    end = time.time()
                    print("Spent computation time:", end-start)
                    print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
                    # print("Result RE:", repr(k), "Verified by FAdo:", is_solution(repr(k), examples, membership2))
                    print("Result RE:", repr(k))
                    finished = True
                    break


            if repr(new_elem) == '0|1' or new_elem.type == Type.CHAR:
                checker = True
            else:
                checker = False

            if checker and is_pdead(k, examples):
                #print(repr(k), "is pdead")
                continue

            if (new_elem.type == Type.K or new_elem.type==Type.Q or checker) and is_ndead(k, examples):
                #print(repr(k), "is ndead")
                continue

            if args.normalform == 'alpha' and k.alpha():
                # print(repr(k), "is not alpha normal form")
                continue

            if args.normalform == 'scnf' and is_not_scnf(k, new_elem):
                # print(repr(k), "is not scnf")
                continue

            if args.redundant and is_redundant(k, examples, new_elem):
                # print(repr(k), "is redundant")
                continue


            w.put((k.getCost(), k))


    if i % 1000 == 0:
        print("Iteration:", i, "\tCost:", cost, "\tScanned REs:", len(scanned), "\tQueue Size:", w.qsize(), "\tTraversed:", traversed)
        end = time.time()
        print("Spent computation time:", end - start)
    i = i+1

print("--end--")
print("count = ")
print(i)
print("answer = " + answer)




