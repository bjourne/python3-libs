# Copyright (C) 2018-2019 Björn Lindqvist <bjourne@gmail.com>
#
# Based on:
# https://tadeuzagallo.com/blog/writing-a-lambda-calculus-interpreter-in-javascript/
from collections import namedtuple
from re import findall

LAMBDA, DOT, LPAREN, RPAREN, LCID, EOF = range(6)

Token = namedtuple('Token', ['type', 'value'])
Ident = namedtuple('Ident', ['id'])
Appl = namedtuple('Appl', ['lhs', 'rhs'])
Abst = namedtuple('Abst', ['id', 'body'])

def term(toks):
    if toks[0].type == LAMBDA:
        toks.pop(0)
        type, value = toks.pop(0)
        assert type == LCID
        assert toks.pop(0).type == DOT
        return Abst(value, term(toks))
    return appl(toks)

def atom(toks):
    peek_type = toks[0].type
    if peek_type == LPAREN:
        toks.pop(0)
        trm = term(toks)
        assert toks.pop(0).type == RPAREN
        return trm
    elif peek_type == LCID:
        return Ident(toks.pop(0).value)
    elif peek_type == LAMBDA:
        toks.pop(0)
        type, value = toks.pop(0)
        assert type == LCID
        assert toks.pop(0).type == DOT
        return Abst(value, term(toks))
    return None

def appl(toks):
    lhs = atom(toks)
    while True:
        rhs = atom(toks)
        if not rhs:
            return lhs
        lhs = Appl(lhs, rhs)

def parse(s):
    types = {'λ' : LAMBDA,
             '\\' : LAMBDA,
             '.' : DOT,
             '(' : LPAREN,
             ')' : RPAREN}
    parts = findall(r'(\(|\)|λ|\\|[a-z][a-z]*|\.)', s)
    toks = [Token(types.get(p, LCID), p) for p in parts]
    toks.append(Token(EOF, None))

    result = term(toks)
    assert toks[0].type == EOF
    return result

def format(ast, brackets = False, inleft = False):
    appl_fmt = '%s %s'
    abst_fmt = r'\%s. %s'
    if brackets:
        appl_fmt = '(%s)' % appl_fmt
        abst_fmt = '(%s)' % abst_fmt
    if isinstance(ast, Appl):
        lhs = format(ast.lhs, brackets, True)
        rhs = format(ast.rhs, brackets, inleft)
        if isinstance(ast.rhs, Appl) and not brackets:
            return '%s (%s)' % (lhs, rhs)
        return appl_fmt % (lhs, rhs)
    if isinstance(ast, Abst):
        params = [ast.id]
        body = ast.body
        while isinstance(body, Abst):
            params += [body.id]
            body = body.body
        body = format(body, brackets, False)
        param_str = ' '.join(params)
        if inleft and not brackets:
            return r'(\%s. %s)' % (param_str, body)
        return abst_fmt % (param_str, body)
    return ast.id

def is_value(ast):
    return isinstance(ast, Abst) or isinstance(ast, Ident)

def subst(id, e, arg):
    if isinstance(e, Abst):
        return Abst(e.id, subst(id, e.body, arg))
    elif isinstance(e, Appl):
        return Appl(subst(id, e.lhs, arg), subst(id, e.rhs, arg))
    elif isinstance(e, Ident):
        if e.id == id:
            return arg
        return e

def step(ast):
    if isinstance(ast, Appl):
        lhs, rhs = ast
        if isinstance(lhs, Abst) and is_value(rhs):
            return subst(lhs.id, lhs.body, rhs)

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

def test_step():
    examples = [
        (r'(\x. x) y', 'y'),
        (r'(\x. z) y', 'z'),
        (r'(\x. x x) y', 'y y'),
        (r'(\x. (\y. k)) k', '\y. k')
        ]
    for inp, out in examples:
        expr = parse(inp)
        expr = step(expr)
        assert format(expr) == out

if __name__ == '__main__':
    test_format()
    test_step()
