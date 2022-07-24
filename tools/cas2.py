# This is some annoyingly complicated code.
from ast import *
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from sys import exit

def parse_expr(expr):
    return parse(expr).body[0].value

def call(n, args):
    return Call(Name(n), args, [])

def const(v):
    return Constant(v)

def commut_join(id, args):
    flat = []
    for a in args:
        if type(a) == Call and a.func.id == id:
            flat.extend(a.args)
        else:
            flat.append(a)
    return call(id, flat)

def mul(args):
    return commut_join('mul', args)

def add(args):
    return commut_join('add', args)

def pow(l, r):
    if type(l) == Call:
        id = l.func.id
        if id == 'pow':
            # Not sure about this (probably wrong).
            return pow(l.args[0], mul([l.args[1], r]))
    return call('pow', [l, r])

def div(l, r):
    return mul([l, pow(r, const(-1))])

def build(tree):
    tp = type(tree)
    if tp == BinOp:
        tp_op = type(tree.op)
        l = build(tree.left)
        r = build(tree.right)
        if tp_op == Add:
            return add([l, r])
        elif tp_op == Sub:
            return add([l, mul([const(-1), r])])
        elif tp_op == Mult:
            return mul([l, r])
        elif tp_op == Pow:
            return pow(l, r)
        elif tp_op == Div:
            return div(l, r)
        assert False
    elif tp == Call:
        id = tree.func.id
        if id == 'sqrt':
            return pow(tree.args[0], const(Fraction(1, 2)))
        return call(id, [build(a) for a in tree.args])
    elif tp == UnaryOp:
        return mul([const(-1), build(tree.operand)])
    return tree

def fold_add(args):
    coeff = 0
    counts = defaultdict(int)
    for arg in args:
        tp = type(arg)
        if tp == Constant:
            coeff += arg.value
        else:
            if tp == Name:
                key = arg
                value = 1
            elif tp == Call and arg.func.id == 'mul':
                mul_args = arg.args
                head, tail = mul_args[0], mul_args[1:]
                if type(head) == Constant:
                    key = mul(tail)
                    value = head.value
                else:
                    key = mul(mul_args)
                    value = 1
            else:
                assert False
            key = unparse(key)
            counts[key] += value
    counts = sorted(counts.items())
    head = const(coeff)
    tail = [mul([const(v), parse_expr(k)]) for (k, v) in counts]
    tail = [fold(e) for e in tail]
    if not tail:
        return head
    if coeff == 0:
        if len(tail) == 1:
            return tail[0]
        return add(tail)
    return add([head] + tail)

def fold_mul(args):
    counts = defaultdict(lambda: add([]))
    coeff = 1
    for arg in args:
        tp = type(arg)
        if tp == Constant:
            coeff *= arg.value
        else:
            if tp == Name:
                key = arg
                value = const(1)
            elif tp == Call:
                id = arg.func.id
                if id == 'pow':
                    key = arg.args[0]
                    value = arg.args[1]
                else:
                    key = arg
                    value = const(1)
            else:
                assert False
            key = unparse(key)
            counts[key].args.append(value)

    counts = sorted(counts.items())
    head = const(coeff)
    tail = [pow(parse_expr(k), v) for (k, v) in counts]
    tail = [fold(e) for e in tail]
    if not tail:
        return head
    elif coeff == 1:
        if len(tail) == 1:
            return tail[0]
        return mul(tail)
    elif coeff == 0:
        return const(0)
    return mul([head] + tail)

def fold_pow(args):
    base, power = args
    tp_base = type(base)
    tp_power = type(power)
    if tp_power == Constant:
        power_val = power.value
        if tp_base == Constant:
            base_val = base.value
            if int(power_val) == power_val:
                v = base_val**abs(power_val)
                if power_val < 0:
                    v = Fraction(1, v)
                return const(v)
            if base_val == -1 and power_val == Fraction(1, 2):
                return Name('I')
        elif power_val == 1:
            return base
        elif power_val == 0:
            return const(1)
    if tp_base == Constant:
        if base.value == 1:
            return const(1)
    elif tp_base == Call and base.func.id == 'mul':
        tree = mul([pow(a, power) for a in base.args])
        return fold(tree)
    return pow(base, power)

def fold_usub(tree):
    tp = type(tree)
    return Constant(-tree.value)

def fold(tree):
    FOLDERS = {
        'add' : fold_add,
        'mul' : fold_mul,
        'pow' : fold_pow
    }
    tp = type(tree)
    if tp == Call:
        id = tree.func.id
        folder = FOLDERS.get(id, lambda x: call(id, x))
        return folder([fold(a) for a in tree.args])
    elif tp == UnaryOp:
        return fold_usub(tree.operand)
    return tree

# All this shit mostly to get negative numbers to look ok.
def is_negative(tree):
    if type(tree) == Call and tree.func.id == 'mul':
        return tree.args[0].value < 0
    return False

def tidy(tree, negate):
    tp = type(tree)
    if tp == Call and tree.func.id == 'add':
        ret = tidy(tree.args[0], False)
        for arg in tree.args[1:]:
            negate = is_negative(arg)
            op = Sub() if negate else Add()
            ret = BinOp(ret, op, tidy(arg, negate))
        return ret
    elif tp == Call and tree.func.id == 'mul':
        args = tree.args
        ret = tidy(args.pop(0), False)
        if type(ret) == Constant and ret.value < 0:
            if negate:
                ret.value = -ret.value
                if ret.value == 1:
                    ret = tidy(args.pop(0), False)
            elif ret.value == -1:
                ret = UnaryOp(USub(), args.pop(0))
        for arg in args:
            ret = BinOp(ret, Mult(), tidy(arg, False))
        return ret
    elif tp == Call and tree.func.id == 'pow':
        base = tidy(tree.args[0], False)
        if type(base) == Constant and base.value < 0:
            base = UnaryOp(USub(), const(-base.value))
        power = tree.args[1]
        if type(power) == Constant and power.value == Fraction(1, 2):
            return call('sqrt', [base])
        power = tidy(power, False)
        return BinOp(base, Pow(), power)
    elif tp == Constant:
        v = tree.value
        if type(v) == Fraction:
            return BinOp(const(v.numerator),
                         Div(),
                         const(v.denominator))
    return tree

def prettify(expr):
    return expr.replace(' * ', '').replace(' ** ', '^')

def check(output, expected):
    if output != expected:
        print('got     : %s' % (output,))
        print('expected: %s' % (expected,))
        assert False

def test_build():
    exprs = [
        ('(-x)*(-(y*-z))', 'mul(-1, x, -1, y, -1, z)'),
        ('x+x*y+z', 'add(x, mul(x, y), z)'),
        ('x*(a+b+c)', 'mul(x, add(a, b, c))'),
        ('(a+b+c)**2', 'pow(add(a, b, c), 2)'),
        ('(-2)**3', 'pow(mul(-1, 2), 3)'),
        ('5/3', 'mul(5, pow(3, -1))'),
        ('3*x - 10*x', 'add(mul(3, x), mul(-1, 10, x))'),
        ('x**(1/2)*x**2', 'mul(pow(x, mul(1, pow(2, -1))), pow(x, 2))'),
        ('4**(1/2)', 'pow(4, mul(1, pow(2, -1)))'),
        ('(x**y)**z', 'pow(x, mul(y, z))'),
        ('3**(x+y)+3**(y+x)', 'add(pow(3, add(x, y)), pow(3, add(y, x)))'),
        ('cos(x+y)+cos(y+x)', 'add(cos(add(x, y)), cos(add(y, x)))'),
        ('cos(-5*(x+y))', 'cos(mul(-1, 5, add(x, y)))'),
        ('(x+4)**n*(x+4)**1', 'mul(pow(add(x, 4), n), pow(add(x, 4), 1))'),
        ('(x+4)**n*(x+4)', 'mul(pow(add(x, 4), n), add(x, 4))')
    ]
    for input, expected in exprs:
        tree = parse_expr(input)
        tree = build(tree)
        check(unparse(tree), expected)

# Folding results in simplified expressions
def test_fold():
    exprs = [
        # Constants
        ('10**2*3', '300'),
        ('2**(1/2)', 'pow(2, Fraction(1, 2))'),
        ('(2**(1/2))**2', '2'),
        ('2**(-1)', 'Fraction(1, 2)'),

        ('1**(1/2)', '1'),

        # One var, linear exprs
        ('5*x-5*x', '0'),
        ('x*(3 + 4)', 'mul(7, x)'),
        ('-(-(-(x)))', 'mul(-1, x)'),
        ('x + 0', 'x'),

        # Multiple vars, linear exprs
        ('x+x*y+z', 'add(mul(x, y), x, z)'),

        # roots
        ('x**(1/2)*x**(1/2)', 'x'),
        ('sqrt(x)**2', 'x'),
        ('sqrt(x)*sqrt(x)', 'x'),

        # Integer powers
        ('x**1*x**(-1)', '1'),
        ('x**1', 'x'),
        ('(a + b + c)**0', '1'),
        ('(2*x + 3*x)**2', 'mul(25, pow(x, 2))'),
        ('-3*x*8*x', 'mul(-24, pow(x, 2))'),
        ('3*x**2 - 10*x**2 + 8', 'add(8, mul(-7, pow(x, 2)))'),
        ('x**2*x', 'pow(x, 3)'),

        # Imaginary numbers
        ('(-1)**(1/2)', 'I'),
        ('(-5)**(1/2)', 'pow(-5, Fraction(1, 2))'),
        ('sqrt(-1)', 'I'),
        ('(-25)**(1/2)', 'pow(-25, Fraction(1, 2))'),

        ('(x+4)**n*(x+4)**2', 'pow(add(4, x), add(2, n))'),
        ('(x+4)**n*(x+4)**1', 'pow(add(4, x), add(1, n))'),
        ('(x+4)**n*(x+4)', 'pow(add(4, x), add(1, n))'),
        ('(x+y)*(y+x)', 'pow(add(x, y), 2)'),
        ('(x+y+z)*(y+x+z)*(z+x+y)', 'pow(add(x, y, z), 3)'),
        ('cos(x-y)*cos(x-y)', 'pow(cos(add(mul(-1, y), x)), 2)')
    ]
    for input, expected in exprs:
        tree = parse_expr(input)
        tree = build(tree)
        tree = fold(tree)
        check(unparse(tree), expected)

def test_tidy():
    exprs = [
        ('2**(1/2)', 'sqrt(2)'),
        ('-1*x', '-x'),
        ('-(-(-(x)))', '-x'),
        ('8 - w*75*x*y', '8 - 75wxy'),
        ('3*x**2 - 10*x**2 + 8', '8 - 7x^2'),
        ('(-5)**(1/8)', '(-5)^(1 / 8)'),
        ('4**(1/2)', 'sqrt(4)'),
        ('(x**y)**z', 'x^(yz)')
    ]
    for input, expected in exprs:
        tree = parse_expr(input)
        tree = build(tree)
        tree = fold(tree)
        tree = tidy(tree, False)
        output = prettify(unparse(tree))
        check(output, expected)

test_build()
test_fold()
test_tidy()
