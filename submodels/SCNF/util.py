import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodels', 'fado')))
from submodels.fado.FAdo.cfg import *
from .parsetree import *

import copy

def membership(regex, string):
    return bool(re.fullmatch(regex, string))


def is_solution(regex, examples, membership, prefix_for_neg_test=None, suffix_for_neg_test=None):
    if regex == '@emptyset':
        return False

    for string in examples.getPos():
        if not membership(regex, string):
            return False

    if prefix_for_neg_test is not None:
        regex = '(' + prefix_for_neg_test + ')' + '(' + regex + ')'
    if suffix_for_neg_test is not None:
        regex = '(' + regex + ')' + '(' + suffix_for_neg_test + ')'

    for string in examples.getNeg():
        if membership(regex, string):
            return False

    return True


def is_pdead(s, examples, alphabet_size=5):
    s_spreadAll = s.spreadAll(alphabet_size)

    for string in examples.getPos():
        if not membership(s_spreadAll, string):
            return True
    return False


def is_ndead(s, examples, prefix=None, suffix=None):
    regex = s.spreadNP()

    if regex == '@emptyset':
        return False

    if prefix:
        regex = prefix + regex
    if suffix:
        regex = regex + suffix

    for string in examples.getNeg():
        if membership(regex, string):
            return True

    return False


def is_not_scnf(s, new_elem, alphabet_size=5):
    all_char = [Character(str(x)) for x in range(alphabet_size)]

    if repr(new_elem) == str(Or(*all_char)) or new_elem.type == Type.CHAR:
        checker = True
    else:
        checker = False

    if (new_elem.type == Type.K or new_elem.type == Type.Q) and s.starnormalform():
        # print(repr(k), "starNormalForm")
        return True

    if checker and s.redundant_concat1():
        # print("concat1")
        return True

    if s.redundant_concat2(alphabet_size):
        # print("concat2")
        return True

    if checker and s.KCK(alphabet_size):
        # print(repr(k), "is kc_qc")
        return True

    if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and s.KCQ(alphabet_size):
        # print(repr(k), "KCQ")
        return True

    if checker and s.QC():
        # print(repr(k), "is kc_qc")
        return True

    if new_elem.type == Type.Q and s.OQ():
        # print(repr(k), "is OQ")
        return True

    if checker and s.orinclusive(alphabet_size):
        # print(repr(k), "is orinclusive")
        return True

    if checker and s.prefix():
        # print(repr(k), "is prefix")
        return True

    if (new_elem.type == Type.K or new_elem.type == Type.Q or checker) and s.sigmastar(alphabet_size):
        # print(repr(k), "is equivalent_KO")
        return True
    return False


def is_redundant(s, examples, new_elem, alphabet_size):
    all_char = [Character(str(x)) for x in range(alphabet_size)]
    if repr(new_elem) == str(Or(*all_char)) or new_elem.type == Type.CHAR:
        checker = True
    else:
        checker = False
    if not (new_elem.type == Type.Q or checker):
        return False

    # unroll
    unrolled_state = copy.deepcopy(s)
    unrolled_state.prior_unroll()
    tmp = unrolled_state.reprAlpha2(alphabet_size)
    unsp = list(i.replace('#', '({})*'.format(Or(*all_char))) for _, i in tmp)

    # check part
    for state in unsp:
        is_red = True
        for string in examples.getPos():
            if membership(state, string):
                is_red = False
                break
        if is_red:
            return True
    return False



