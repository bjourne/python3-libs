# Copyright (C) 2018-2019 Björn Lindqvist <bjourne@gmail.com>
#
# Based on:
# https://tadeuzagallo.com/blog/writing-a-lambda-calculus-interpreter-in-javascript/
from collections import namedtuple
from re import findall
from string import ascii_lowercase

LAM, DOT, LPAREN, RPAREN, LCID, EOF = range(6)

Token = namedtuple('Token', ['type', 'value'])
Ident = namedtuple('Ident', ['id'])
Appl = namedtuple('Appl', ['lhs', 'rhs'])
Abst = namedtuple('Abst', ['id', 'body'])

def abst(toks):
    toks.pop(0)
    params = []
    while toks[0].type != DOT:
        type, value = toks.pop(0)
        assert type == LCID
        params.append(value)
    toks.pop(0)
    abst = Abst(params.pop(), term(toks))
    while params:
        abst = Abst(params.pop(), abst)
    return abst

def term(toks):
    if toks[0].type == LAM:
        return abst(toks)
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
    elif peek_type == LAM:
        return abst(toks)
    return None

def appl(toks):
    lhs = atom(toks)
    while True:
        rhs = atom(toks)
        if not rhs:
            return lhs
        lhs = Appl(lhs, rhs)

def parse(s):
    types = {'λ' : LAM,
             '\\' : LAM,
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

def freevars(e):
    if isinstance(e, Abst):
        return freevars(e.body) - set([e.id])
    if isinstance(e, Appl):
        return freevars(e.lhs) | freevars(e.rhs)
    return set([e.id])

def rename(e, f, t):
    if isinstance(e, Abst):
        return Abst(e.id, rename(e.body, f, t))
    if isinstance(e, Appl):
        return Appl(rename(e.lhs, f, t), rename(e.rhs, f, t))
    return Ident(t if e.id == f else e.id)

def newid(id, ast):
    if id in freevars(ast):
        n = len(ascii_lowercase)
        i = (ascii_lowercase.find(id) + 1) % n
        return newid(ascii_lowercase[i], ast)
    return id

def subst(id, e, arg):
    if isinstance(e, Abst):
        if e.id == id:
            # Won't subst bound variable
            return e
        ren_id = newid(e.id, arg)
        ren_body = rename(e.body, e.id, ren_id)
        return Abst(ren_id, subst(id, ren_body, arg))
    elif isinstance(e, Appl):
        return Appl(subst(id, e.lhs, arg), subst(id, e.rhs, arg))
    if e.id == id:
        return arg
    return e

def step(e):
    if isinstance(e, Appl):
        lhs, rhs = e
        # Do stuff with the left branch if possible, otherwise work on
        # the right side.
        if isinstance(lhs, Abst):
            return subst(lhs.id, lhs.body, rhs)
        elif isinstance(lhs, Appl):
            lhs = step(lhs)
            return Appl(lhs, rhs) if lhs else None
        rhs = step(rhs)
        return Appl(lhs, rhs) if rhs else None
    elif isinstance(e, Abst):
        body = step(e.body)
        return Abst(e.id, body) if body else None
    return None

def eval(e, verbose = False):
    while True:
        if verbose:
            print(format(e))
        e2 = step(e)
        if not e2:
            break
        if e == e2:
            if verbose:
                print('Infinite recursion detected!')
            break
        e = e2
    return e

if __name__ == '__main__':
    expr = parse(r'(\p. p (\t f. f) (\t f. t)) (\t f. t)')
    print(format(eval(expr, verbose = True)))
    #eval(expr, verbose = True)


    # expr = parse(r'(\z. z) ((\s. \z. s z) s z)')
    # print(format(step(expr)))
    #eval(expr, verbose = True)
