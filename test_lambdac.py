# Copyright (C) 2018-2019 Björn Lindqvist <bjourne@gmail.com>
from lambdac import format, parse, step

# See https://www.easycalculation.com/analytical/lambda-calculus.php
def test_format():
    assert format(parse(r'a (\b. a) c')) == r'a (\b. a) c'
    assert format(parse(r'(\b. a) c')) == r'(\b. a) c'
    assert format(parse('(a b) (c d)')) == 'a b (c d)'
    assert format(parse('a (b c)')) == 'a (b c)'
    str3 = r'\x. \y. z \m. o'
    assert format(parse(str3)) == r'\x y. z \m. o'
    assert format(parse(str3), brackets = True) \
        == r'(\x y. (z (\m. o)))'

    s = 'x y z'
    assert format(parse(s), brackets = True) == '((x y) z)'
    assert format(parse(s)) == 'x y z'

    examples = [
        (r'(\b. \c. b c (\t. \f. f)) (\t. \f. f)',
         r'(\b c. b c \t f. f) \t f. f'),
        (r'(\n. \s. \z. s (n s z)) (\s. \z. s z)',
         r'(\n s z. s (n s z)) \s z. s z'),
        (r'(\x. x) (\y. y)',
         r'(\x. x) \y. y'),
        (r'((a (\x. x)) a)', r'a (\x. x) a')
        ]
    for inp, out in examples:
        assert format(parse(inp)) == out

def test_parsing():
    # Support for parsing abstraction contraction
    expr = parse(r'\x y z. x')
    assert expr.id == 'x'
    assert expr.body.id == 'y'
    assert expr.body.body.id == 'z'
    assert expr.body.body.body.id == 'x'

    examples = [
        (r'(\n. \s. \z. s (n s z)) (\s. \z. z)',
         r'(\n s z. s (n s z)) \s z. z'),
        (r'(\n s z. s (n s z)) (\s z. z)',
         r'(\n s z. s (n s z)) \s z. z'),
        ]
    for inp, out in examples:
        assert format(parse(inp)) == out


def test_step():
    examples = [
        (r'(\x. x) y', 'y'),
        (r'(\x. z) y', 'z'),
        (r'(\x. x x) y', 'y y'),
        (r'(\x. (\y. k)) k', r'\y. k')
        ]
    for inp, out in examples:
        expr = parse(inp)
        expr = step(expr)
        assert format(expr) == out

if __name__ == '__main__':
    test_format()
    test_step()
    test_parsing()
