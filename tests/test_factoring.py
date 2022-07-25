# Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
from fractions import Fraction
from tools.factoring import factorize_rat

def test_factoring():
    examples = [
        (-2, [(-1, 1), (2, 1)]),
        (Fraction(1, 8), [(2, -3)]),
        (-1, [(-1, 1)]),
        (0, [(0, 1)]),
        (1, [(1, 1)]),
        (2, [(2, 1)]),
        (-100, [(-1, 1), (2, 2), (5, 2)]),
        (Fraction(1, 2), [(2, -1)]),
        (-Fraction(1, 2), [(-1, 1), (2, -1)]),
    ]
    for input, expected in examples:
        assert factorize_rat(input) == expected
