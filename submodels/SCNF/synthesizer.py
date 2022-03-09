from queue import PriorityQueue
from .util import *
from .examples import *

import copy


def synthesis(examples, count_limit=50000, start_with_no_concat=False, prefix_for_neg_test=None,
              suffix_for_neg_test=None, alphabet_size=5):
    w = PriorityQueue()
    scanned = set()
    w.put((REGEX().getCost(), REGEX()))

    i = 0
    traversed = 1

    answer = None
    finished = False

    while not w.empty() and not finished and i < count_limit:
        tmp = w.get()
        s = tmp[1]
        cost = tmp[0]

        hasHole = s.hasHole()

        all_char = [Character(str(x)) for x in range(alphabet_size)]
        start_elems = get_start_elem(all_char, start_with_no_concat, is_first=(i == 0))

        # print("state : ", s, " cost: ", cost)
        if hasHole:
            for j, new_elem in enumerate(start_elems):

                k = copy.deepcopy(s)

                if not k.spread(new_elem):
                    continue

                traversed += 1
                if repr(k) in scanned:
                    continue
                else:
                    scanned.add(repr(k))

                if not k.hasHole():
                    if is_solution(repr(k), examples, membership, prefix_for_neg_test, suffix_for_neg_test):
                        answer = k
                        finished = True
                        break

                if repr(new_elem) == str(Or(*all_char)) or new_elem.type == Type.CHAR:
                    checker = True
                else:
                    checker = False

                if checker and is_pdead(k, examples, alphabet_size):
                    continue

                if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and is_ndead(k, examples,
                                                                                                prefix_for_neg_test,
                                                                                                suffix_for_neg_test):
                    continue

                if is_not_scnf(k, new_elem, alphabet_size):
                    continue

                if is_redundant(k, examples, new_elem, alphabet_size):
                    continue

                w.put((k.getCost(), k))
        i = i + 1

    return answer


def get_start_elem(all_char, start_with_no_concat, is_first):
    start_elems = [] + all_char + [Or(), Or(*all_char)]
    if not is_first or not start_with_no_concat:
        start_elems.append(Concatenate(Hole(), Hole()))
    start_elems += [Question(), KleenStar()]

    return start_elems


def main():
    regex = synthesis(Examples(pos=set(['222222', '2222', '22', '222222222']), neg=set(
        ['003120310', '214244', '02420021', '0204001', '021431', '1024', '1124', '222423442', '3212', '1133233'])),
                      2000, start_with_no_concat=False)
    print(regex)


if __name__ == "__main__":
    main()
