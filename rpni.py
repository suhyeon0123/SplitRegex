import FAdo.fa as fa
import itertools

STATE_UNKNOWN = 1
STATE_ACCEPT = 2
STATE_REJECT = 3

class Argmax():
    def __init__(self):
        self.k = None
        self.v = None

    def add_item(self, k, v):
        assert v is not None
        if self.v is None:
            self.k = k
            self.v = v
        elif self.v < v:
            self.k = k
            self.v = v

    def get_argmax(self):
        return self.k

    def __len__(self):
        return 0 if self.v is None else 1


def compatible(M, p, q):
    if M[p] == STATE_UNKNOWN or M[q] == STATE_UNKNOWN:
        return True
    return M[p] == M[q]


def make_trie(pos, neg):
    A = fa.DFA()
    M = dict()
    A.setInitial(A.addState(''))

    for s in list(pos) + list(neg):
        state = ''
        for c in s:
            A.addTransition(
                    A.stateIndex(state),
                    c,
                     A.stateIndex(state + c, autoCreate=True))
            state += c

    for s in A.States:
        if s in pos:
            M[s] = STATE_ACCEPT
        elif s in neg:
            M[s] = STATE_REJECT
        else:
            M[s] = None

    return A, M


def merge(A, M, p, q, visited=list()):
    """
    A: fa.DFA
    M: dict[state->unk/acc/rej]
    p, q: states
    """

    if p == q:
        return A, M

    if (p, q) in visited:
        return A, M

    if M[p] == STATE_ACCEPT and M[q] == STATE_REJECT:
        return False, False
    if M[p] == STATE_REJECT and M[q] == STATE_ACCEPT:
        return False, False

    visited.append((p, q))

    Aret = A.dup()
    Mret = M.copy()

    if Mret.get(p, STATE_UNKNOWN) == STATE_UNKNOWN:
        Mret[p] = Mret.get(q, STATE_UNKNOWN)

    for s in Aret.Sigma:
        np = Aret.Delta(Aret.stateIndex(p), s)
        nq = Aret.Delta(Aret.stateIndex(q), s)

        if np is not None and nq is not None and np != nq:
            Aret, Mret = merge(Aret, Mret, Aret.States[np], Aret.States[nq], visited)
            if Aret is False:
                visited.pop()
                return False, False

    # subst q to p
    pp = Aret.stateIndex(p)
    qq = Aret.stateIndex(q)

    # copy q's outgoing to p
    if qq in Aret.delta:
        for c in Aret.delta[qq]:
            Aret.delta[pp][c] = Aret.delta[qq][c]

    # copy q's incoming to p
    if qq == A.Initial:
        A.setInitial(pp)

    for x in Aret.delta:
        for c in Aret.delta[x]:
            if Aret.delta[x][c] == qq:
                Aret.delta[x][c] = pp

    visited.pop()
    return Aret, Mret


def mergible(A, M, p, q, visited=list()):
    if p == q: return True
    if (p, q) in visited: return True

    if not compatible(M, p, q):
        return False

    visited.append((p, q))

    for s in A.Sigma:
        np = A.Delta(A.stateIndex(p), s)
        nq = A.Delta(A.stateIndex(q), s)

        if np is not None and nq is not None:
            if not mergible(A, M, A.States[np], A.States[nq], visited):
                visited.pop()
                return False

    visited.pop()
    return True


def equivScore(A, M, p, q, visited=set()):
    s = 0

    assert p != q

    if (p, q) in visited:
        return 0

    visited.add((p, q))

    if (M[p] == STATE_ACCEPT and M[q] == STATE_ACCEPT):
        s += 1
    if (M[p] == STATE_REJECT and M[q] == STATE_REJECT):
        s += 1

    for c in A.Sigma:
        np = A.Delta(A.stateIndex(p), c)
        nq = A.Delta(A.stateIndex(q), c)

        if np is not None and nq is not None:
            s += equivScore(A, M, A.States[np], A.States[nq], visited)

    return s


def depth(A, p):
    queue = [(A.States[A.Initial], 0)]
    visited = set([queue[0][0]])

    while len(queue) > 0:
        q, d = queue.pop(0)

        if q == p:
            return d
        else:
            for s in A.Sigma:
                nq = A.Delta(A.stateIndex(q), s)
                if nq is not None and A.States[nq] not in visited:
                    queue.append((A.States[nq], d + 1))
                    visited.add(A.States[nq])

    assert False


def computeScore(A, M, p, q):
    return (equivScore(A, M, p, q), -depth(A, p))


def pred_notnone(x):
    return x is not None


def blue_fringe(pos, neg):
    A, M = make_trie(pos, neg)

    red = set([''])
    blue = set(A.States[p]
            for p in filter(
                pred_notnone, 
                (A.Delta(A.stateIndex(r), s)
                    for r, s in itertools.product(red, A.Sigma)))) - red

    score = Argmax()
    visited = set()

    merged = None
    while len(blue) > 0:
        for q in blue:
            merged = False
            for p in red:
                if (p, q) in visited:
                    merged = True
                elif mergible(A, M, p, q) and p != q:
                    score.add_item((p, q), computeScore(A, M, p, q))
                    visited.add((p, q))
                    merged = True

            if not merged:
                red.add(q)
                break

        if merged:
            (p, q) = score.get_argmax()
            A, M = merge(A, M, p, q)
            score = Argmax()
            visited = set()

        blue = set(A.States[p]
                for p in filter(
                    pred_notnone, 
                    (A.Delta(A.stateIndex(r), s)
                        for r, s in itertools.product(red, A.Sigma)))) - red

    for s in A.States:
        if M.get(s, STATE_UNKNOWN) == STATE_ACCEPT:
            A.addFinal(A.stateIndex(s))

    A = A.trim()
    A = A.complete()
    return A

class REPR_FADO_REGEX():
    def __init__(self, s):
        self.s = str(s).replace(' ', '')

    def __repr__(self):
        return self.s


def synthesis(examples, *args, **kwargs):
    A = blue_fringe(examples.pos, examples.neg)
    return REPR_FADO_REGEX(A.regexp())


if __name__ == '__main__':

    class Ex:
        def __init__(self, pos, neg):
            assert len(set(pos) & set(neg)) == 0
            self.pos = set(pos)
            self.neg = set(neg)

    def FA_run(A, w):
        st = A.Initial
        for c in w:
            st = A.Delta(st, c)
        return st

    def test(pos, neg):
        A = blue_fringe(pos, neg)

        for s in pos:
            st = FA_run(A, s)
            if st not in A.Final:
                print(s, 'failed; expected: positive')

        for s in neg:
            st = FA_run(A, s)
            if st in A.Final:
                print(s, 'failed; expected: negative')

        print(repr(synthesis(Ex(pos, neg))))

    test(['0', '01', '010', '0101', '01010'],
         ['1', '10', '101', '1010', '10101'])

    test(['0', '00', '000', '000000', '00000'],
         ['1', '11', '111', '1111', '11111'])
