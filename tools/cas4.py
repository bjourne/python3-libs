from ast import *
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce
from math import sqrt
from sys import exit

def call(n, args):
    return Call(Name(n), args, [])

def const(v):
    return Constant(v)

CONST_0 = const(0)
CONST_1 = const(1)
CONST_N1 = const(-1)
CONST_HALF = const(Fraction(1, 2))

def parse_expr(expr):
    return parse(expr).body[0].value

def is_constant(tree):
    tp = type(tree)
    if tp == Constant:
        return True
    elif tp == Name:
        return False
    elif tp == Call:
        return all(is_constant(a) for a in tree.args)
    elif tp == UnaryOp:
        return is_constant(tree.operand)
    assert False

def split_factors(tree):
    tp = type(tree)
    if tp == Call:
        id = tree.func.id
        args = tree.args
        if id in {'mul', 'add'}:
            is_consts = [(is_constant(a), a) for a in args]
            consts = [a for (c, a) in is_consts if c]
            non_consts = [a for (c, a) in is_consts if not c]
            return consts, non_consts
        elif id == 'pow':
            if all(is_constant(a) for a in args):
                return [tree], []
            return [], [tree]
    elif tp == Constant:
        return [tree], []
    elif tp == Name:
        return [], [tree]
    assert False

def commute_join(args, fun):
    flat = []
    for a in args:
        if type(a) == Call and a.func.id == fun:
            flat.extend(a.args)
        else:
            flat.append(a)
    return flat

def filter_commute(args, fun, id_el):
    args = [a for a in args
            if type(a) != Constant or a.value != id_el]
    if len(args) == 0:
        return const(id_el)
    elif len(args) == 1:
        return args[0]
    return call(fun, args)

def const_add(args):
    rat_sum = 0
    terms = []
    for a in args:
        if type(a) == Constant:
            rat_sum += a.value
        else:
            terms.append(a)
    terms.insert(0, const(rat_sum))
    return terms

def nonconst_add(args):
    kvs = defaultdict(list)
    for arg in args:
        consts, non_consts = split_factors(arg)
        key = tuple(sorted(unparse(nc) for nc in non_consts))
        kvs[key].append(mul(consts))
    terms = []
    for k, v in kvs.items():
        term = mul([add(v)] + [parse_expr(nc) for nc in k])
        terms.append(term)
    return terms

def add(args):
    args = commute_join(args, 'add')
    args = sorted(args, key = unparse)

    # This is how we break the recursive loop.
    if all(is_constant(a) for a in args):
        terms = const_add(args)
    else:
        terms = nonconst_add(args)

    return filter_commute(terms, 'add', 0)

def mul(args):
    args = commute_join(args, 'mul')
    kvs = defaultdict(list)
    coeff = 1
    for arg in args:
        if type(arg) == Constant:
            v = arg.value
            coeff *= v
        else:
            b, e = arg, const(1)
            if type(arg) == Call and arg.func.id == 'pow':
                b, e = arg.args[0], arg.args[1]
            kvs[unparse(b)].append(e)

    if coeff == 0:
        return CONST_0

    factors = [const(coeff)]
    for b, e in kvs.items():
        factor = pow([parse_expr(b), add(e)])
        factors.append(factor)

    return filter_commute(factors, 'mul', 1)

def div(args):
    return mul([args[0], pow([args[1], CONST_N1])])

def sub(args):
    return add([args[0], mul([CONST_N1, args[1]])])

def pow(args):
    l, r = args
    tp_l = type(l)
    tp_r = type(r)
    if tp_l == Call:
        id_l = l.func.id
        if id_l == 'pow':
            return pow([l.args[0], mul([l.args[1], r])])
        if id_l == 'mul':
            return mul([pow([a, r]) for a in l.args])

    if tp_r == Constant:
        r_val = r.value
        if r_val >= 0:
            if r_val == 0:
                return CONST_1
            elif r_val == 1:
                return l
            elif tp_l == Constant:
                l_val = l.value
                if l_val >= 0:
                    res = l_val ** r_val
                    if res == int(res):
                        return const(int(res))
        else:
            if tp_l == Constant and r.value == -1:
                return const(Fraction(1, l.value))
    return call('pow', args)

MERGERS = {Add : add, Sub: sub, Mult : mul, Pow : pow, Div : div}

def merge_int(tree):
    tp = type(tree)
    if tp == BinOp:
        merger = MERGERS[type(tree.op)]
        args = [merge(tree.left), merge(tree.right)]
        return merger(args)
    elif tp == Constant:
        return const(tree.value)
    elif tp == Name:
        return tree
    elif tp == UnaryOp:
        return mul([CONST_N1, merge(tree.operand)])
    elif tp == Call:
        id = tree.func.id
        args = [merge(a) for a in tree.args]
        if id == 'sqrt':
            return pow([args[0], CONST_HALF])
        else:
            return call(id, args)
    assert False

def merge(tree):
    ret = merge_int(tree)
    print('%-40s => %-40s' % (unparse(tree), unparse(ret)))
    return ret

def check(output, expected):
    if output != expected:
        print('got     : %s' % (output,))
        print('expected: %s' % (expected,))
        assert False

def test_const():
    examples = [
        (0, '0'),
        (1, '1'),
        (Fraction(1, 2), 'Fraction(1, 2)'),
        (-10, '-10'),
        (4, '4'),
    ]
    for input, expected in examples:
        output = unparse(const(input))
        check(output, expected)

def test():
    exprs = [
        # Constants
        ('4', '4'),
        ('2**2', '4'),

        ('1/2', 'Fraction(1, 2)'),
        ('2*3*4', '24'),
        ('10+20', '30'),
        ('-2**2', '-4'),

        ('2**(1/2)', 'pow(2, Fraction(1, 2))'),
        ('10**2*3', '300'),
        ('(2**(1/2))**2', '2'),
        ('2**(-1)', 'Fraction(1, 2)'),
        ('1**(1/2)', '1'),

        ('(-2)*(-2)', '4'),

        # Expansion of roots
        ('sqrt(4)', '2'),

        # Single variables, no exps
        ('y*y*y + 1 + 10', 'add(11, pow(y, 3))'),
        ('x - x', '0'),
        ('5*x-5*x', '0'),
        ('x*(3 + 4)', 'mul(7, x)'),
        ('-(-(-(x)))', 'mul(-1, x)'),
        ('x + 0', 'x'),

        # No distribution by default
        ('x*(y + z)', 'mul(x, add(y, z))'),

        # Multiple variables
        ('x*y*z', 'mul(x, y, z)'),
        ('x*(y*z)', 'mul(x, y, z)'),
        ('x+x*y+z', 'add(mul(x, y), x, z)'),
        ('(x+y)*(y+x)', 'pow(add(x, y), 2)'),

        # Roots
        ('sqrt(2)*x + 3*x', 'mul(add(3, pow(2, Fraction(1, 2))), x)'),
        ('x**(1/2)*x**(1/2)', 'x'),
        ('sqrt(x)**2', 'x'),
        ('sqrt(x)*sqrt(x)', 'x'),

        ('sqrt(-1)*sqrt(-1)', '-1'),

        # Trigonometry
        ('cos(-x)', 'cos(mul(-1, x))'),

        # Simple exponents
        ('x**1*x**(-1)', '1'),
        ('x**1', 'x'),
        ('(a + b + c)**0', '1'),

        # Pow should distribute over mul
        ('(2*x + 3*x)**2', 'mul(25, pow(x, 2))'),
        ('sqrt((2*x + 3*x)**2)', 'mul(5, x)'),

        ('-3*x*8*x', 'mul(-24, pow(x, 2))'),
        ('3*x**2 - 10*x**2 + 8', 'add(8, mul(-7, pow(x, 2)))'),
        ('x**2*x', 'pow(x, 3)'),
        ('x**(-z)*x**(z + 1)', 'x'),

        # Difficult exponents
        ('x**(n+1)*x**n', 'pow(x, add(1, mul(2, n)))'),
        ('x**(n+2)*x**(n+3)', 'pow(x, add(5, mul(2, n)))'),
        ('x**(-z)', 'pow(x, mul(-1, z))'),
        ('x**(-z)*x', 'pow(x, add(1, mul(-1, z)))'),
        ('x**(-z)*x**(z)', '1'),
        ('x**(-z)*x**(z + 1)', 'x'),
        ('((x**y)**z)**w', 'pow(x, mul(y, z, w))'),
        ('((x**y)**z)**(w/y)', 'pow(x, mul(z, w))'),

        # Subtraction
        ('-z+z-1', '-1'),

        # Random
        ('(x+4)**n*(x+4)**2', 'pow(add(4, x), add(2, n))'),
        ('(x+4)**n*(x+4)**1', 'pow(add(4, x), add(1, n))'),
        ('(x+4)**n*(x+4)', 'pow(add(4, x), add(1, n))'),
        ('(x+y+z)*(y+x+z)*(z+x+y)', 'pow(add(x, y, z), 3)'),
        ('cos(x-y)*cos(x-y)', 'pow(cos(add(mul(-1, y), x)), 2)'),
    ]
    for input, expected in exprs:
        tree = parse_expr(input)
        tree = merge(tree)
        output = unparse(tree)
        check(output, expected)

test_const()
test()
