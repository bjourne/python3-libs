# Copyright (C) 2022 Björn A. Lindqvist <bjourne@gmail.com>
from collections import Counter
from fractions import Fraction
from math import sqrt

def factorize_int(n):
    if n < 0:
        yield -1
    n = abs(n)
    j = 2
    factors = []
    while n > 1:
        for i in range(j, int(sqrt(n + 0.05)) + 1):
            if n % i == 0:
                n //= i
                j = i
                yield i
                break
        else:
            if n > 1:
                yield n
            break

def factorize_rat(rat):
    if rat in (-1, 0, 1):
        return [(rat, 1)]
    rat = Fraction(rat)
    p, q = rat.numerator, rat.denominator
    fs = []
    if p != 1:
        fs.extend(Counter(factorize_int(p)).items())
    if q != 1:
        fs.extend((b, -e) for (b, e) in Counter(factorize_int(q)).items())
    return fs
