# Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
#
# A simple computer algebra system capable of finding first and second
# degree polynomials' roots. It relies on the AST modules so that I
# don't have to write my own parser.
import sys
from ast import *
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce
from math import sqrt

def order_terms(mv):
    def orderfun(term):
        powers, c = term
        deg = -max(p for n, p in powers)
        names = ''.join(n for n, p in powers)
        return names
    return list(sorted(mv.items(), key = orderfun))

def parse_mv(mv):
    def pow(n, e):
        return Name(n) if e == 1 else BinOp(Name(n), Pow(), Constant(e))
    def term(es, c, first = False):
        fs = [pow(n, e) for n, e in es]
        fs = [Constant(abs(c))] + fs if abs(c) != 1 else fs
        ret = UnaryOp(USub(), fs[0]) if first and c < 0 else fs[0]
        return reduce(lambda x, y: BinOp(x, Mult(), y), fs[1:], ret)
    mv = [t for t in order_terms(mv) if t[1] != 0]
    ret = term(*mv[0], first = True) if mv else Constant(0)
    join = {True : Add(), False : Sub()}
    return reduce(lambda ret, el: BinOp(ret, join[el[1] > 0], term(*el)),
                  mv[1:], ret)

def parse_expr(expr):
    return parse(expr).body[0].value

def prettify(expr):
    return expr.replace(' * ', '').replace(' ** ', '^')

def elwise(mv1, mv2, fun):
    combos = list(mv1) + list(mv2)
    return defaultdict(int, {c : fun(mv1[c], mv2[c]) for c in combos})

def add(mv1, mv2):
    return elwise(mv1, mv2, lambda x, y: x + y)

def sub(mv1, mv2):
    return elwise(mv1, mv2, lambda x, y: x - y)

def pow(mv1, mv2):
    items = list(mv2.items())
    pows, n = items[0]
    err = 'Non-negative integer powers only'
    assert len(items) == 1 and pows == () and n > 0, err
    mv3 = defaultdict(int, {() : 1})
    return reduce(lambda x, y: mul(x, mv1), range(n), mv3)

def mul(mv1, mv2):
    mv3 = defaultdict(int)
    for es1, c1 in mv1.items():
        for es2, c2 in mv2.items():
            es3 = Counter(dict(es1))
            es3.update(dict(es2))
            es3 = tuple(sorted(es3.items()))
            mv3[es3] += c1*c2
    return mv3

def eval(tree):
    tp = type(tree)
    if tp == BinOp:
        BINOPS = {Add : add, Sub : sub, Mult : mul, Pow : pow}
        l, r = eval(tree.left), eval(tree.right)
        return BINOPS[type(tree.op)](l, r)
    elif tp == Name:
        return defaultdict(int, {((tree.id, 1),) : 1})
    elif tp == Constant:
        return defaultdict(int, {() : tree.value})
    elif tp == UnaryOp:
        return mul(eval(tree.operand), defaultdict(int, {() : -1}))

def parse_fraction(f):
    p, q = f.numerator, f.denominator
    if q != 1:
        if p != 0:
            return BinOp(Constant(p), Div(), Constant(q))
        return Constant(0)
    return Constant(p)

def roots_1st(cs):
    f = Fraction(-cs[0], cs[1])
    return [parse_fraction(f)]

def roots_2nd(cs):
    def make_root(b, a, tree, op):
        tree = BinOp(Constant(-b), op, tree)
        return BinOp(tree, Div(), Constant(2*a))

    d = cs[1]**2 - 4*cs[2]*cs[0]
    ad = abs(d)
    sqrt_ad = sqrt(ad)
    int_sqrt_ad = int(sqrt_ad)

    simple = True
    if sqrt_ad == int_sqrt_ad:
        rhs = Constant(int_sqrt_ad)
    else:
        rhs = Call(Name('sqrt'), [Constant(ad)], [])
        simple = False

    if d < 0:
        rhs = BinOp(Name('i'), Mult(), rhs)
        simple = False

    if not simple:
        x1 = make_root(cs[1], cs[2], rhs, Add())
        x2 = make_root(cs[1], cs[2], rhs, Sub())
    else:
        x1 = Fraction(-cs[1] + int_sqrt_ad, 2 * cs[2])
        x2 = Fraction(-cs[1] - int_sqrt_ad, 2 * cs[2])
        x1 = parse_fraction(x1)
        x2 = parse_fraction(x2)
    return [x1, x2]

def roots(mv):
    deg = max([max([e[1] for e in k], default = 0)
               for k in mv if mv[k]], default = 0)
    assert 1 <= deg <= 2, 'Degree 1 or 2 only'
    by_var = defaultdict(lambda: defaultdict(int))
    for var_pows, c in mv.items():
        for v, p in var_pows:
            by_var[v][p] = c
    by_var = list(by_var.items())
    assert len(by_var) <= 1, 'Univariate polynomials only'
    name, cs = by_var[0]
    cs[0] = mv[()]

    rs = roots_1st(cs) if deg == 1 else roots_2nd(cs)
    rs = [Compare(Name(name), [Eq()], [r]) for r in rs]

    return BoolOp(Or(), rs)

def main():
    tree = parse(sys.argv[2]).body[0].value
    if type(tree) == Compare:
        tree = BinOp(tree.left, Sub(), tree.comparators[0])
    mv = eval(tree)
    print('==>', unparse(roots(mv) if sys.argv[1] == 'solve' else parse_mv(mv)))

if __name__ == '__main__':
    main()
