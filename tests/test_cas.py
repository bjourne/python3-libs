# Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
from ast import BinOp, Sub, parse, unparse
from tools.cas import (eval, order_terms, parse_expr,
                       parse_mv, prettify, roots)

def test_order_terms():
    terms = [({
        (('x', 1), ('z', 1)): 1,
        (('w', 1), ('x', 1)): -1,
        (('y', 1), ('z', 1)): 1,
        (('w', 1), ('y', 1)): -1
    }, [
        ((('w', 1), ('x', 1)), -1),
        ((('w', 1), ('y', 1)), -1),
        ((('x', 1), ('z', 1)), 1),
        ((('y', 1), ('z', 1)), 1)
    ])]
    for input, expected in terms:
        output = order_terms(input)
        assert output == expected

def normalize_and_prettify(expr):
    return prettify(unparse(parse_mv(eval(parse_expr(expr)))))

def test_normalize():
    exprs = [
        ('(x+y)*(z-w)', '-wx - wy + xz + yz'),
        ('(x+y)*y', 'xy + y^2')
    ]
    for input, expected in exprs:
        output = normalize_and_prettify(input)
        assert output == expected

def test_roots():
    exprs = [
        ('3*x**2 == -x**2 + 5*x', 'x == 5 / 4 or x == 0'),
        ('x**2 == 9', 'x == 3 or x == -3'),
        ('x**2 + x == 9', 'x == (-1 + sqrt(37)) / 2 or x == (-1 - sqrt(37)) / 2'),
        ('x**2 + 4*x == -30',
         'x == (-4 + i * sqrt(104)) / 2 or x == (-4 - i * sqrt(104)) / 2'),
        ('4*x == 9', 'x == 9 / 4')
    ]
    for input, expected in exprs:
        tree = parse_expr(input)
        tree = BinOp(tree.left, Sub(), tree.comparators[0])
        mv = eval(tree)
        output = unparse(roots(mv))
        assert output == expected
