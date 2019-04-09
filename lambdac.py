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

def to_string(ast, brackets = False, inleft = False):
    appl_fmt = '%s %s'
    abst_fmt = r'\%s. %s'
    if brackets:
        appl_fmt = '(%s)' % appl_fmt
        abst_fmt = '(%s)' % abst_fmt
    if isinstance(ast, Appl):
        lhs = to_string(ast.lhs, brackets, True)
        rhs = to_string(ast.rhs, brackets, inleft)
        if isinstance(ast.rhs, Appl) and not brackets:
            return '%s (%s)' % (lhs, rhs)
        return appl_fmt % (lhs, rhs)
    if isinstance(ast, Abst):
        params = [ast.id]
        body = ast.body
        while isinstance(body, Abst):
            params += [body.id]
            body = body.body
        body = to_string(body, brackets, False)
        param_str = ' '.join(params)
        if inleft and not brackets:
            return r'(\%s. %s)' % (param_str, body)
        return abst_fmt % (param_str, body)
    return ast.id

# See https://www.easycalculation.com/analytical/lambda-calculus.php
def test_to_string():
    assert to_string(parse(r'a (\b. a) c')) == r'a (\b. a) c'
    assert to_string(parse(r'(\b. a) c')) == r'(\b. a) c'
    assert to_string(parse('(a b) (c d)')) == 'a b (c d)'
    assert to_string(parse('a (b c)')) == 'a (b c)'
    str3 = r'\x. \y. z \m. o'
    assert to_string(parse(str3)) == r'\x y. z \m. o'
    assert to_string(parse(str3), brackets = True) \
        == r'(\x y. (z (\m. o)))'

    s = 'x y z'
    assert to_string(parse(s), brackets = True) == '((x y) z)'
    assert to_string(parse(s)) == 'x y z'

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
        assert to_string(parse(inp)) == out

def is_value(ast):
    return isinstance(ast, Abst) or isinstance(ast, Ident)

def eval(ast):
    while True:
        if isinstance(ast, Appl):
            if is_value(ast.lhs) and is_value(ast.rhs):
                ast = subst(...)
            elif is_value(ast.lhs):
                ast.rhs = eval(ast.rhs)
            else:
                ast.lhs = eval(ast.lhs)
        else:
            return ast

if __name__ == '__main__':
    test_to_string()
    expr = parse(r'(\x. y) a')
    eval(expr)
