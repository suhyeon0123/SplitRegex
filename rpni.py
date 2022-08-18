# Wojciech Wieczorek,
# Grammatical inference: algorithms, routines and applications,
# pp. 23--27,
# Springer, 2017.

import FAdo.fa as fa
import FAdo.reex as reex
import FAdo.conversions as conv

import itertools
import logging
logger = logging.Logger(__name__)

STATE_UNKNOWN = 0
STATE_ACCEPT = 1
STATE_REJECT = -1

REGEX_EMPTYSET = '@emptyset'
REGEX_EPSILON = '@epsilon'

STATE_DEAD = '@DeaD'


def alphabet(S):
    return set(a for a in itertools.chain.from_iterable(w for w in S))


def prefixes(S):
    result = set()
    for s in S:
        for i in range(len(s) + 1):
            result.add(s[:i])
    return result


def suffixes(S):
    result = set()
    for s in S:
        for i in range(len(s) + 1):
            result.add(s[i:])
    return result


def substrings(S):
    return prefixes(suffixes(S))


def cat(A, B):
    return set(a + b for a in A for b in B)


def ql(S):
    return sorted(S, key=lambda x: (len(x), x))


def buildPTA(S):
    A = fa.DFA()
    q = {u: A.addState(u) for u in prefixes(S)}

    for w in iter(q):
        u, a = w[:-1], w[-1:]
        if a != '':
            A.addTransition(q[u], a, q[w])
        if w in S:
            A.addFinal(q[w])
    A.setInitial(q[''])

    return A


def merge(q1, q2, A):
    A.mergeStates(q2, q1)
    return A


def run(w, q, A):
    for c in w:
        q = None if q is None else A.Delta(q, c)
    return q


def accepts(w, q, A):
    return run(w, q, A) in A.Final


def buildTable(pos, neg):
    OT = dict()
    EXP = suffixes(pos | neg)
    Red = {''}
    Blue = alphabet(pos | neg)
    for p in Red | Blue:
        for e in EXP:
            if p + e in pos:
                OT[p, e] = STATE_ACCEPT
            elif p + e in neg:
                OT[p, e] = STATE_REJECT
            else:
                OT[p, e] = STATE_UNKNOWN
    return Red, Blue, EXP, OT


def compatible(r, b, EXP, OT):
    return not any((OT[r, e] == STATE_ACCEPT and OT[b, e] == STATE_REJECT) or
                   (OT[r, e] == STATE_REJECT and OT[b, e] == STATE_ACCEPT)
                   for e in EXP)


def fillHoles(Red, Blue, EXP, OT):
    for b in ql(Blue):
        found = False
        for r in ql(Red):
            if compatible(r, b, EXP, OT):
                found = True
                for e in EXP:
                    if OT[b, e] != STATE_UNKNOWN:
                        OT[r, e] = OT[b, e]
        if not found:
            logger.info('{}: incompatible: {}'.format(__name__, b))
            return False

    for r in Red:
        for e in EXP:
            if OT[r, e] == STATE_UNKNOWN:
                OT[r, e] = STATE_ACCEPT

    for b in ql(Blue):
        found = False
        for r in ql(Red):
            if compatible(r, b, EXP, OT):
                found = True
                for e in EXP:
                    if OT[b, e] == STATE_UNKNOWN:
                        OT[b, e] = OT[r, e]
        if not found:
            logger.info('{}: incompatible: {}'.format(__name__, b))
            return False
    return True


def buildFA(Red, Blue, EXP, OT):
    A = fa.NFA()
    A.setSigma(alphabet(Red | Blue | EXP))
    q = dict()

    for r in Red:
        q[r] = A.addState(r)

    for w in Red | Blue:
        for e in EXP:
            if w + e in Red and OT[w, e] == STATE_ACCEPT:
                A.addFinal(q[w + e])

    for w in q:
        for u in q:
            for a in A.Sigma:
                if all(OT[u, e] == OT[w + a, e] for e in EXP):
                    A.addTransition(q[w], a, q[u])
    A.addInitial(q[''])

    A.trim()

    return A


def distinguishable(u, v, EXP, OT):
    return any(
        OT[u, e] in {STATE_ACCEPT, STATE_REJECT}
        and OT[v, e] in {STATE_ACCEPT, STATE_REJECT}
        and OT[u, e] != OT[v, e]
        for e in EXP)


def rpni(pos, neg, count_limit=None):
    Red, Blue, EXP, OT = buildTable(pos, neg)
    Sigma = alphabet(pos | neg)

    x = ql(b for b in Blue if all(distinguishable(b, r, EXP, OT) for r in Red))
    iter_count = 0

    while x and (count_limit is None or iter_count < count_limit):
        iter_count += 1
        Red.add(x[0])
        Blue.discard(x[0])
        Blue.update(cat({x[0]}, Sigma))

        for u in Blue:
            for e in EXP:
                if u + e in pos:
                    OT[u, e] = STATE_ACCEPT
                elif u + e in neg:
                    OT[u, e] = STATE_REJECT
                else:
                    OT[u, e] = STATE_UNKNOWN

        x = ql(b for b in Blue if all(
            distinguishable(b, r, EXP, OT) for r in Red))

    if not fillHoles(Red, Blue, EXP, OT):
        logger.info('Cannot fill empty holes')
        A = buildPTA(pos)
    else:
        A = buildFA(Red, Blue, EXP, OT)
        if not (all(A.evalWordP(w)
                    for w in pos) and not any(A.evalWordP(w) for w in neg)):
            logger.info('FA test failed')
            A = buildPTA(pos)

    A.setSigma(Sigma)
    return A


class REPR():
    def __init__(self, rex):
        self.rex = rex

    def __repr__(self):
        return self.rex

    def __str__(self):
        return self.rex


def product(A, B):
    X = fa.DFA()
    q = dict()

    X.setSigma(A.Sigma | B.Sigma)

    q = {(a, b): X.addState((a, b))
         for a, _ in enumerate(A.States) for b, _ in enumerate(B.States)}

    for a, _ in enumerate(A.States):
        for b, _ in enumerate(B.States):
            for c in X.Sigma:
                na = A.Delta(a, c)
                nb = B.Delta(b, c)
                if na is not None and nb is not None:
                    X.addTransition(q[a, b], c, q[na, nb])

    return X


def quotient(lang, pref, suff):
    A = buildPTA(lang).toDFA().trim().complete()
    prefA = reex.str2regexp(pref).toDFA().trim().complete()
    suffA = reex.str2regexp(suff).toDFA().trim().complete()

    PQ = product(prefA, A)
    SQ = product(A, suffA).toNFA()

    PQ.setInitial(PQ.stateIndex((prefA.Initial, A.Initial)))
    for i, x in enumerate(PQ.States):
        if x[0] in prefA.Final:
            PQ.addFinal(i)

    SQ.setInitial(
        set(SQ.stateIndex((q, suffA.Initial)) for q in range(len(A.States))))
    for i, x in enumerate(SQ.States):
        if x[0] in A.Final and x[1] in suffA.Final:
            SQ.addFinal(i)

    PQ.trim()
    SQ.trim()

    X = A.toNFA()

    X.setInitial(set())
    X.setFinal(set())

    for x in PQ.States:
        if x[0] in prefA.Final and A.States[x[1]] != STATE_DEAD:
            X.addInitial(x[1])

    for x in SQ.States:
        if A.States[x[0]] != STATE_DEAD and x[1] == A.Initial:
            X.addFinal(x[0])

    X.trim()

    return set(x for x in substrings(lang) if X.evalWordP(x))


def rpni_regex(pos, neg_raw, count_limit=None, pref='', suff=''):
    if not pref:
        pref = REGEX_EPSILON
    if not suff:
        suff = REGEX_EPSILON

    neg = quotient(neg_raw, pref=pref, suff=suff)
    neg -= pos

    return rpni(pos, neg, count_limit=count_limit)

def synthesis(examples,
              count_limit=None,
              prefix_for_neg_test='',
              suffix_for_neg_test='',
              *args,
              **kwargs):
    A = rpni_regex(examples.pos, examples.neg, count_limit,
                   prefix_for_neg_test, suffix_for_neg_test)
    # return REPR(str(A.reCG()).replace(' ', '').replace('+', '|'))
    return REPR(str(conv.FA2regexpCG_nn(A)).replace(' ', '').replace('+', '|'))
    


import unittest

class Test(unittest.TestCase):
    def check(self, A, pos, neg):
        # logger.info(str(A.reCG()).replace(' ', '').replace('+', '|'))
        logger.info(str(conv.FA2regexpCG_nn(A)).replace(' ', '').replace('+', '|'))
        for w in pos:
            with self.subTest(w=w):
                self.assertTrue(A.evalWordP(w))
        for w in neg:
            with self.subTest(w=w):
                self.assertFalse(A.evalWordP(w))

    def test_case_0(self):
        pos = set(['83', '6666', '834', '366666', '566', '4', '666', '66', '266666', '8666'])
        neg = set(['4566', '0377949', '127075', '76697261', '823292', '5309927380', '48151502', '085', '706', '565786'])
        A = rpni_regex(pos, neg)
        self.check(A, pos, neg)
