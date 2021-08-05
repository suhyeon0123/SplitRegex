from enum import Enum
import random
import re


class Type(Enum):
    HOLE = 0
    CHAR = 1
    K = 2
    Q = 3
    C = 4
    U = 5
    REGEX = 10


class RE:
    def __lt__(self, other):
        return False

    def rpn(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.rpn() + 1
        elif self.type == Type.C or self.type == Type.U:
            return sum(list(i.rpn() for i in self.list)) + len(self.list) - 1
        elif self.type == Type.REGEX:
            return self.r.rpn()
        else:
            return 1

    def make_child(self, count):
        if self.type == Type.K or self.type == Type.Q:
            if self.r.type == Type.HOLE:
                while True:
                    x = get_rand_re(count)
                    if x.type != Type.K and x.type != Type.Q:
                        self.r = x
                        break
            else:
                self.r.make_child(count + 1)
        elif self.type == Type.C or self.type == Type.U:
            for index, regex in enumerate(self.list):
                if regex.type == Type.HOLE:
                    self.list[index] = get_rand_re(count)
                else:
                    self.list[index].make_child(count + 1)
        elif self.type == Type.REGEX:
            if self.r.type == Type.HOLE:
                self.r = get_rand_re(count)
            else:
                self.r.make_child(count + 1)

    def spreadRand(self):
        if self.type == Type.K or self.type == Type.Q:
            if self.r.type == Type.HOLE:
                self.r = rand_char()
            else:
                self.r.spreadRand()
        elif self.type == Type.C or self.type == Type.U:
            for index, regex in enumerate(self.list):
                if regex.type == Type.HOLE:
                    self.list[index] = rand_char()
                else:
                    self.list[index].spreadRand()
        elif self.type == Type.REGEX:
            if self.r.type == Type.HOLE:
                self.r = rand_char()
            else:
                self.r.spreadRand()

    # Pruning Rules
    def starnormalform(self):
        if self.type == Type.K or self.type == Type.Q:
            if self.r.hasEps():
                return True
            else:
                return self.r.starnormalform()
        elif self.type == Type.C or self.type == Type.U:
            return any(list(i.starnormalform() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.starnormalform()

    def hasEps(self):
        if self.type == Type.K or self.type == Type.Q:
            return True
        elif self.type == Type.C:
            return all(list(i.hasEps() for i in self.list))
        elif self.type == Type.U:
            return any(list(i.hasEps() for i in self.list))
        else:
            return False

    def allHole(self):
        if self.type == Type.K or self.type == Type.Q:
            if self.r.allHole():
                return True
            else:
                return False
        elif self.type == Type.C or self.type == Type.U:
            return all(list(i.allHole() for i in self.list))
        elif self.type == Type.HOLE:
            return True
        else:
            return False

    def redundant_concat1(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.redundant_concat1()

        elif self.type == Type.C:
            for index, regex in enumerate(self.list):

                if regex.type == Type.K:
                    # x*x
                    if index + 1 < len(self.list) and (regex.r.type == Type.CHAR or regex.r.type == Type.U) and repr(
                            regex.r) == repr(self.list[index + 1]):
                        return True
                    elif regex.r.type == Type.C and index + len(regex.r.list) < len(self.list) and not self.list[
                        index + len(regex.r.list)].hasHole():
                        tmp = Concatenate()
                        tmp.list = self.list[index + 1:index + len(regex.r.list) + 1]
                        if repr(regex.r) == repr(tmp):
                            return True

                    # x*x?
                    elif index + 1 < len(self.list) and self.list[index + 1].type == Type.Q and repr(regex.r) == repr(
                            self.list[index + 1].r):
                        return True

                elif regex.type == Type.Q:
                    # x?x
                    if index + 1 < len(self.list) and (regex.r.type == Type.CHAR or regex.r.type == Type.U) and repr(
                            regex.r) == repr(self.list[index + 1]):
                        return True
                    elif regex.r.type == Type.C and index + len(regex.r.list) < len(self.list) and not self.list[
                        index + len(regex.r.list)].hasHole():
                        tmp = Concatenate()
                        tmp.list = self.list[index + 1:index + len(regex.r.list) + 1]
                        if repr(regex.r) == repr(tmp):
                            return True
                    # x?x*
                    elif index + 1 < len(self.list) and self.list[index + 1].type == Type.K and repr(regex.r) == repr(
                            self.list[index + 1].r):
                        return True

            return any(list(i.redundant_concat1() for i in self.list))

        elif self.type == Type.U:
            return any(list(i.redundant_concat1() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.redundant_concat1()
        else:
            return False

    def redundant_concat2(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.redundant_concat2()

        elif self.type == Type.C:
            for index, regex in enumerate(self.list):

                if regex.type == Type.K:
                    for index2, regex2 in enumerate(self.list):

                        tmp = Concatenate()
                        if index - index2 == 1 or index2 - index == 1:
                            if regex2.hasEps() and is_inclusive(regex, regex2):
                                return True
                        elif index > index2:
                            tmp.list = self.list[index2:index]
                            if all(list(i.hasEps() for i in tmp.list)) and is_inclusive(regex, tmp):
                                return True
                        elif index < index2:
                            tmp.list = self.list[index + 1:index2 + 1]
                            if all(list(i.hasEps() for i in tmp.list)) and is_inclusive(regex, tmp):
                                return True
                        else:
                            continue

            return any(list(i.redundant_concat2() for i in self.list))

        elif self.type == Type.U:
            return any(list(i.redundant_concat2() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.redundant_concat2()
        else:
            return False

    def KCK(self):
        if self.type == Type.K:
            if self.r.type == Type.C:

                for index, regex in enumerate(self.r.list):
                    if regex.type == Type.K and not regex.hasHole():
                        left = False
                        right = False
                        if index == 0:
                            left = True
                        elif index == 1:
                            tmp1 = self.r.list[0]
                        else:
                            tmp1 = Concatenate()
                            tmp1.list = self.r.list[0:index]

                        if index == len(self.r.list) - 1:
                            right = True
                        elif index == len(self.r.list) - 2:
                            tmp2 = self.r.list[len(self.r.list) - 1]
                        else:
                            tmp2 = Concatenate()
                            tmp2.list = self.r.list[index + 1:len(self.r.list)]

                        if not regex.r.hasEps() and (left or is_inclusive(regex, tmp1)) and (
                                right or is_inclusive(regex, tmp2)):
                            return True

            return self.r.KCK()

        elif self.type == Type.Q:
            return self.r.KCK()

        elif self.type == Type.C or self.type == Type.U:
            return any(list(i.KCK() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.KCK()
        else:
            return False

    def KCQ(self):
        if self.type == Type.K:
            if self.r.type == Type.C:

                # single symbol
                for index, regex in enumerate(self.r.list):

                    if not regex.hasHole():
                        left = False
                        right = False

                        if index == 0:
                            left = True
                            lefteps = True
                        elif index == 1:
                            tmp1 = self.r.list[0]
                            lefteps = tmp1.hasEps()
                        else:
                            tmp1 = Concatenate()
                            tmp1.list = self.r.list[0:index]
                            lefteps = all(list(i.hasEps() for i in tmp1.list))

                        if index == len(self.r.list) - 1:
                            right = True
                            righteps = True
                        elif index == len(self.r.list) - 2:
                            tmp2 = self.r.list[len(self.r.list) - 1]
                            righteps = tmp2.hasEps()
                        else:
                            tmp2 = Concatenate()
                            tmp2.list = self.r.list[index + 1:len(self.r.list)]
                            righteps = all(list(i.hasEps() for i in tmp2.list))

                        if lefteps and righteps and (left or is_inclusive(KleenStar(regex), tmp1)) and (
                                right or is_inclusive(KleenStar(regex), tmp2)):
                            return True

                # single regex
                for i in range(len(self.r.list) - 2):
                    for j in range(i + 2, len(self.r.list)):
                        if (i == 0 and j == len(self.r.list)) or j > len(self.r.list):
                            continue
                        regex = Concatenate()
                        regex.list = self.r.list[i:j]

                        if not regex.hasHole():
                            left = False
                            right = False

                            if i == 0:
                                left = True
                                lefteps = True
                            elif i == 1:
                                tmp1 = self.r.list[0]
                                lefteps = tmp1.hasEps()
                            else:
                                tmp1 = Concatenate()
                                tmp1.list = self.r.list[0:i]
                                lefteps = all(list(r.hasEps() for r in tmp1.list))

                            if j == len(self.r.list):
                                right = True
                                righteps = True
                            elif j == len(self.r.list) - 1:
                                tmp2 = self.r.list[len(self.r.list) - 1]
                                righteps = tmp2.hasEps()
                            else:
                                tmp2 = Concatenate()
                                tmp2.list = self.r.list[j + 1:len(self.r.list)]
                                righteps = all(list(i.hasEps() for i in tmp2.list))

                            if lefteps and righteps and (left or is_inclusive(KleenStar(regex), tmp1)) and (
                                    right or is_inclusive(KleenStar(regex), tmp2)):
                                return True

            return self.r.KCQ()

        elif self.type == Type.Q:
            return self.r.KCQ()

        elif self.type == Type.C or self.type == Type.U:
            return any(list(i.KCQ() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.KCQ()
        else:
            return False

    def QC(self):
        if self.type == Type.K:
            return self.r.QC()

        elif self.type == Type.Q:
            # (xx?)? (xx*)?
            if self.r.type == Type.C and (self.r.list[len(self.r.list) - 1].type == Type.K or self.r.list[
                len(self.r.list) - 1].type == Type.Q):
                if len(self.r.list) == 2:
                    tmp = self.r.list[0]
                else:
                    tmp = Concatenate()
                    tmp.list = self.r.list[0:len(self.r.list) - 1]
                if repr(tmp) == repr(self.r.list[len(self.r.list) - 1].r):
                    return True

            return self.r.QC()

        elif self.type == Type.C or self.type == Type.U:
            return any(list(i.QC() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.QC()
        else:
            return False

    def OQ(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.OQ()
        elif self.type == Type.C:
            return any(list(i.OQ() for i in self.list))
        elif self.type == Type.U:
            for regex in self.list:
                if regex.type == Type.Q:
                    return True
            return any(list(i.OQ() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.OQ()
        else:
            return False

    def orinclusive(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.orinclusive()
        elif self.type == Type.C:
            return any(list(i.orinclusive() for i in self.list))

        elif self.type == Type.U:
            for index, regex in enumerate(self.list):
                for index2, regex2 in enumerate(self.list):
                    if index < index2 and (is_inclusive(regex, regex2) or is_inclusive(regex2, regex)):
                        return True

            for index, regex in enumerate(self.list):
                if regex.orinclusive():
                    return True
            return False

        elif self.type == Type.REGEX:
            return self.r.orinclusive()
        else:
            return False

    def prefix(self):
        if self.type == Type.K or self.type == Type.Q:
            return self.r.prefix()
        elif self.type == Type.C:
            return any(list(i.prefix() for i in self.list))

        elif self.type == Type.U:
            for index1, regex1 in enumerate(self.list):
                if regex1.type == Type.CHAR:
                    for index2, regex2 in enumerate(self.list):
                        if index1 < index2 and regex2.type == Type.CHAR:
                            if repr(regex1) == repr(regex2):
                                return True
                        elif index1 < index2 and regex2.type == Type.C:
                            if repr(regex1) == repr(regex2.list[0]):
                                return True
                            if repr(regex1) == repr(regex2.list[len(regex2.list) - 1]):
                                return True

                if regex1.type == Type.C:
                    for index2, regex2 in enumerate(self.list):
                        if index1 < index2 and regex2.type == Type.C:
                            if repr(regex1.list[0]) == repr(regex2.list[0]):
                                return True
                            if repr(regex1.list[len(regex1.list) - 1]) == repr(regex2.list[len(regex2.list) - 1]):
                                return True

            return any(list(i.prefix() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.prefix()
        else:
            return False

    def sigmastar(self):
        if self.type == Type.K:
            if repr(self.r) != '0|1':
                return bool(re.fullmatch(repr(self.r), '0')) and bool(re.fullmatch(repr(self.r), '1'))

            return self.r.sigmastar()
        elif self.type == Type.Q:
            return self.r.sigmastar()
        elif self.type == Type.C or self.type == Type.U:
            return any(list(i.sigmastar() for i in self.list))
        elif self.type == Type.REGEX:
            return self.r.sigmastar()
        else:
            return False


# node types
class Hole(RE):
    def __init__(self):
        self.level = 0
        self.type = Type.HOLE

    def __repr__(self):
        return '#'

    def hasHole(self):
        return True


class REGEX(RE):
    def __init__(self, r=Hole()):
        self.r = r
        self.type = Type.REGEX

    def __repr__(self):
        return repr(self.r)

    def repr_labeled(self):
        if self.r.type == Type.C:
            return '{}'.format(self.r.repr_labeled())
        else:
            return '({})'.format(repr(self.r))

    def hasHole(self):
        return self.r.hasHole()


class Character(RE):
    def __init__(self, c):
        self.c = c
        self.level = 0
        self.type = Type.CHAR

    def __repr__(self):
        return self.c

    def hasHole(self):
        return False


class KleenStar(RE):
    def __init__(self, r=Hole()):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
        self.type = Type.K

    def __repr__(self):

        if self.r.level > self.level:
            self.string = '({})*'.format(self.r)
            return self.string
        else:
            self.string = '{}*'.format(self.r)
            return self.string

    def hasHole(self):
        if not self.hasHole2:
            return False

        if not self.r.hasHole():
            self.hasHole2 = False
        return self.hasHole2


class Question(RE):
    def __init__(self, r=Hole()):
        self.r = r
        self.level = 1
        self.string = None
        self.hasHole2 = True
        self.type = Type.Q

    def __repr__(self):

        if self.r.level > self.level:
            self.string = '({})?'.format(self.r)
            return self.string
        else:
            self.string = '{}?'.format(self.r)
            return self.string

    def hasHole(self):
        if not self.hasHole2:
            return False

        if not self.r.hasHole():
            self.hasHole2 = False
        return self.hasHole2


class Concatenate(RE):
    def __init__(self, *regexs):
        self.list = list()
        for regex in regexs:
            self.list.append(regex)
        self.level = 2
        self.string = None
        self.hasHole2 = True
        self.type = Type.C

    def __repr__(self):

        def formatSide(side):
            if side.level > self.level:
                return '({})'.format(side)
            else:
                return '{}'.format(side)

        str_list = []
        for regex in self.list:
            str_list.append(formatSide(regex))
        return ''.join(str_list)

    def repr_labeled(self):

        str_list = []
        for regex in self.list:
            if regex.type == Type.C:
                str_list.append('{}'.format(regex.repr_labeled()))
            else:
                str_list.append('({})'.format(regex))
        return ''.join(str_list)

    def hasHole(self):
        if not self.hasHole2:
            return False

        self.hasHole2 = any(list(i.hasHole() for i in self.list))
        return self.hasHole2


class Or(RE):
    def __init__(self, a=Hole(), b=Hole()):
        self.list = list()
        self.list.append(a)
        self.list.append(b)
        self.level = 3
        self.string = None
        self.hasHole2 = True
        self.type = Type.U

    def __repr__(self):

        def formatSide(side):
            if side.level > self.level:
                return '({})'.format(side)
            else:
                return '{}'.format(side)

        str_list = []
        for regex in self.list:
            if str_list:
                str_list.append("|")
            str_list.append(formatSide(regex))
        return ''.join(str_list)

    def hasHole(self):
        if not self.hasHole2:
            return False

        self.hasHole2 = any(list(i.hasHole() for i in self.list))
        return self.hasHole2


def is_inclusive(superset, subset):
    # R -> R    sup-nohole sub-nohole
    if repr(superset) == repr(subset) and not superset.hasHole():
        return True
    # R -> (0+1)*   sup-nohole sub-hole
    if repr(superset) == '(0|1|2|3|4)*':
        return True
    # made of 0s -> 0*, made of 1s -> 1* - nohole
    if repr(superset) == '0*' and '1' not in repr(subset) and not subset.hasHole():
        return True
    if repr(superset) == '1*' and '0' not in repr(subset) and not subset.hasHole():
        return True
    # R -> R*, R -> R?, R? -> R* - nohole
    if (superset.type == Type.K or superset.type == Type.Q) and not superset.hasHole():
        if repr(superset.r) == repr(subset):
            return True
        elif subset.type == Type.Q and repr(superset.r) == repr(subset.r):
            return True
    # R -> (R + r)*     sub- no hole
    if superset.type == Type.K and superset.r.type == Type.U and not subset.hasHole():
        for index, regex in enumerate(superset.r.list):
            if is_inclusive(KleenStar(regex), subset):
                return True


def get_rand_re(depth):
    case = random.randrange(0, depth)

    if case > 3:
        return rand_char()
    else:
        case = random.randrange(0, 5)
        if case <= 0:
            return Or()
        elif case <= 1:
            return Concatenate(Hole(), Hole())
        elif case <= 2:
            return KleenStar()
        elif case <= 3 and depth != 1:
            return Question()
        else:
            return Hole()
        
def rand_char(alpha_size=5):     
    case = random.randrange(0, alpha_size)
    return Character(str(case))