# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
from programs.lambdac import *

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
        (r'(\x. (\y. k)) k', r'\y. k'),
        (r'(\z. z) ((\s. \z. s z) s z)',
         r'(\s z. s z) s z'),

        # alpha-renaming
        (r'(\f. \y. f y) (\x. y) a', r'(\z. (\x. y) z) a')
        ]
    for inp, out in examples:
        expr = parse(inp)
        expr = step(expr)
        assert format(expr) == out

def test_eval():
    examples = [
        (r'(\f y. f y) (\x. y) q', 'y'),
        (r'(\n s z. s (n s z)) (\s z. z)',
         r'\s z. s z'),
        (r'(\n. \s. \z. s (n s z)) (\s. \z. s z)',
         r'\s z. s (s z)'),
        # and true q => q
        (r'(\b. \c. b c (\t. \f. f)) (\t. \f. t) q', 'q'),
        # succ (succ 0) => 2
        (r'(\n s z. s (n s z)) ((\n s z. s (n s z)) \s z. z)',
         r'\s z. s (s z)'),
        # plus 0 1 => 1
        (r'(\m n s z. m s (n s z)) (\s z. z) (\s z. s z)',
         r'\s z. s z'),
        # plus 1 2 => 3
        (r'(\m n s z. m s (n s z)) (\s z. s z) (\s z. s (s z))',
         r'\s z. s (s (s z))'),
        (r'(((\x. (\y. y)) (\a. a)) (\b. b))',
         r'\b. b'),

        # alpha-renaming
        (r'(\f. (\x. (f x))) (\y. (\x. y))',  r'\x y. x'),

        # not true => false
        (r'(\p. p (\t f. f) (\t f. t)) (\t f. t)', r'\t f. f')
        ]
    for inp, out in examples:
        expr = parse(inp)
        format_out = format(eval(expr))
        if out != format_out:
            print('%s => %s, expected %s' % (inp, format_out, out))
        assert format(eval(expr)) == out

def test_freevars():
    e = parse('a b')
    assert freevars(e) == {'a', 'b'}

def parse_and_eval(expr):
    expr = eval(parse_with_builtins(expr))
    return format_with_builtins(expr)

def test_predefined():
    examples = [
        ('$and $true $false', '$false'),
        ('$or $true $false', '$true'),
        ('$plus $1 $2', '$3'),
        ('$succ $1', '$2'),
        ('$plus $0 $1', '$1'),
        ('$succ ($succ $0)', '$2'),
        ('$not $true', '$false')
    ]
    for inp, out in examples:
        res = parse_and_eval(inp)
        if res != out:
            print('Got %s, expected %s.' % (res, out))
        assert res == out

if __name__ == '__main__':
    test_format()
    test_step()
    test_parsing()
    test_freevars()
    test_predefined()
