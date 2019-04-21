from collections import namedtuple
import re

Ident = namedtuple('Ident', ['id'])
Appl = namedtuple('Appl', ['lhs', 'rhs'])
Abst = namedtuple('Abst', ['id', 'body'])

def abst(t):
    t.pop(0)
    p = t.pop(0)
    t.pop(0)
    return Abst(p, term(t))

def term(t):
    return abst(t) if t[0] == '\\' else appl(t)

def atom(t):
    if t[0] == '(':
        t.pop(0)
        trm = term(t)
        t.pop(0)
        return trm
    elif ord('a') <= ord(t[0][0]) <= ord('z'):
        return Ident(t.pop(0))
    elif t[0] == '\\':
        return abst(t)

def appl(t):
    lhs = atom(t)
    while 1:
        rhs = atom(t)
        if not rhs:
            return lhs
        lhs = Appl(lhs, rhs)

def parse(s):
    return term(re.findall(r'(\(|\)|\\|[a-z]\w*|\.)', s) + ['='])

def fv(e):
    if isinstance(e, Abst):
        return fv(e.body) - set([e.id])
    if isinstance(e, Appl):
        return fv(e.lhs) | fv(e.rhs)
    return set([e.id])

def rename(e, f, t):
    if isinstance(e, Abst):
        return Abst(e.id, rename(e.body, f, t))
    if isinstance(e, Appl):
        return Appl(rename(e.lhs, f, t), rename(e.rhs, f, t))
    return Ident(t if e.id == f else e.id)

def newid(id,e):
    if id in fv(e):
        v=chr(97+(ord(id[0])-96)%26)
        return newid(v,e)
    return id

def subst(id, e, arg):
    if isinstance(e, Abst):
        if e.id == id:
            return e
        r_id = newid(e.id, arg)
        r_body = rename(e.body, e.id, r_id)
        return Abst(r_id, subst(id, r_body, arg))
    elif isinstance(e, Appl):
        return Appl(subst(id, e.lhs, arg), subst(id, e.rhs, arg))
    return arg if e.id == id else e

def step(e):
    if isinstance(e, Appl):
        lhs, rhs = e
        if isinstance(lhs, Abst):
            return subst(lhs.id, lhs.body, rhs)
        elif isinstance(lhs, Appl):
            lhs = step(lhs)
            return Appl(lhs, rhs)
        rhs = step(rhs)
        return Appl(lhs, rhs)
    elif isinstance(e, Abst):
        body = step(e.body)
        return Abst(e.id, body)
    raise RuntimeError('hi')

def E(a):
    try:
        return E(step(a))
    except RuntimeError:
        return a
def F(e):
    if isinstance(e, Abst):
        return r'(\%s. %s)' % (e.id, F(e.body))
    if isinstance(e, Appl):
        return r'(%s %s)' % (F(e.lhs), F(e.rhs))
    return e.id

if __name__ == '__main__':
    expr = parse(r'((\a. (\b. (a (a (a b))))) (\ c. (\ d. (c (c d)))))')
    print(F(E(expr)))
    expr = parse(r'(\f. \y. f y) (\x. y) q')
    print(F(E(expr)))
    expr = parse(r'(\a. a) (\x. x) (\y. y)')
    print(F(E(expr)))
    expr = parse(r'(\hi. hi)')
    print(F(E(expr)))
