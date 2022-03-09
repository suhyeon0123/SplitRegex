from queue import PriorityQueue
from .util_snort import *
from .examples import *

import copy


def synthesis(examples, count_limit=50000, start_with_no_concat=False, prefix_for_neg_test=None,
              suffix_for_neg_test=None, alphabet_size=5, type=None, mapping_table={'A':1,'B':1,'C':1 ,'C':1 ,'D':1 ,'E':1 ,'F':1 ,'G':1 ,'H':1 ,'I':1 ,'J':1 ,'K':1, 'L':1,'M':1,'N':1,'O':1,'P':1}):
    w = PriorityQueue()
    scanned = set()
    w.put((REGEX().getCost(), REGEX()))

    i = 0
    traversed = 1

    answer = None
    finished = False
    candidate = None

    while not w.empty() and not finished and i < count_limit:
        tmp = w.get()
        s = tmp[1]
        cost = tmp[0]

        hasHole = s.hasHole()

        all_char = [Character(str(x)) for x in range(alphabet_size)]
        start_elems = get_start_elem_snort(examples, start_with_no_concat, is_first=(i == 0), mapping_table=mapping_table)
        #print(start_elems)

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
                    _solution, _candidate = is_solution(repr(k), examples, membership, prefix_for_neg_test, suffix_for_neg_test)
                    if _solution:
                        answer = k
                        finished = True
                        break
                    if _candidate:
                        candidate = k

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
                #
                # if is_not_scnf(k, new_elem, alphabet_size):
                #     continue
                #
                if is_redundant(k, examples, new_elem, alphabet_size):
                    #print(k)
                    continue

                if k.redundant_charset():
                    continue

                w.put((k.getCost(), k))
        i = i + 1

    return answer, candidate


def get_start_elem(all_char, start_with_no_concat, is_first):
    start_elems = [] + all_char + [Or(), Or(*all_char)]
    if not is_first or not start_with_no_concat:
        start_elems.append(Concatenate(Hole(), Hole()))
    start_elems += [ Question(), KleenStar()]

    return start_elems

def get_start_elem_snort(example, start_with_no_concat, is_first, mapping_table):

    all_char = [Character(str(x)) for x in mapping_table.keys()]
    char_set = [Character('\d'), Character('\w'), Character('!'), Character('.')]


    start_elems = [] + all_char + [Or()]

    for ch in char_set:
        start_elems.append(ch)

    if not is_first or not start_with_no_concat:
        start_elems.append(Concatenate(Hole(), Hole()))
    start_elems += [Question(), KleenStar()]

    return start_elems


def main():

    regex = synthesis(Examples(pos=set(['ABkvBD', 'Ae!y!5_XUtXBD', 'Ae%QPmo0yBD', 'Af!_!_DtBD', 'A4$7!D', 'Au,kBD', 'ACTHct#5CD', 'AP jID1("ICD', 'AF%:b!!gKBD', 'AX*BD']), neg=set(
        ['A#*XD', 'Au,C5D', 'Af!_!_DtP*', '5f!_!_Dt7D', 'A4"7!P', 'Ae%QPmo0y0D', 'AI*fD', 've!y!5_XUtXTD', 'fA%QPmo0yBD', 'AQkvmD'])),
                      1000000, start_with_no_concat=False, type='snort', mapping_table={'A':'a', 'B':'a', 'C':'a', 'D':'a', })
    print(regex)



if __name__ == "__main__":
    main()
