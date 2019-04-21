from collections import *
import re

A = namedtuple('Appl', ['lhs', 'rhs'])
B = namedtuple('Abst', ['id', 'body'])

def abst(t):
    t.pop(0)
    p = t.pop(0)
    t.pop(0)
    return B(p, term(t))

def term(t):
    return abst(t) if t[0] == '\\' else appl(t)

def atom(t):
    if t[0] == '(':
        t.pop(0)
        trm = term(t)
        t.pop(0)
        return trm
    elif ord('a') <= ord(t[0][0]) <= ord('z'):
        return t.pop(0)
    elif t[0] == '\\':
        return abst(t)

def appl(t):
    lhs = atom(t)
    while 1:
        rhs = atom(t)
        if not rhs:
            return lhs
        lhs = A(lhs, rhs)

def parse(s):
    return term(re.findall(r'(\(|\)|\\|[a-z]\w*|\.)', s) + ['='])

def V(e):
    o=type(e)
    return V(e.body)-{e.id} if o==B else V(e.lhs)|V(e.rhs) if o==A else {e}

def R(e, f, t):
    o=type(e)
    if o==B:
        return o(e.id, R(e.body, f, t))
    if o==A:
        return o(R(e.lhs, f, t), R(e.rhs, f, t))
    return t if e == f else e

def newid(id,e):
    if id in V(e):
        v=chr(97+(ord(id[0])-96)%26)
        return newid(v,e)
    return id

def S(id, e, arg):
    o=type(e)
    if o==B:
        if e.id == id: return e
        rid = newid(e.id, arg)
        rbody = R(e.body, e.id, rid)
        return o(rid, S(id, rbody, arg))
    if o==A:
        return o(S(id, e.lhs, arg), S(id, e.rhs, arg))
    return arg if e==id else e

def T(e):
    o=type(e)
    if o==A:
        lhs, rhs = e
        if type(lhs)==B:
            return S(lhs.id, lhs.body, rhs)
        if type(lhs)==A:
            return A(T(lhs), rhs)
        return o(lhs, T(rhs))
    if o==B:
        return o(e.id, T(e.body))
    raise RuntimeError('hi')

def E(a):
    try:
        return E(T(a))
    except RuntimeError:
        return a
def F(e):
    o=type(e)
    if o==B:
        return r'(\%s. %s)' % (e.id, F(e.body))
    if o==A:
        return r'(%s %s)' % (F(e.lhs), F(e.rhs))
    return e

if __name__ == '__main__':
    expr = parse(r'((\a. (\b. (a (a (a b))))) (\ c. (\ d. (c (c d)))))')
    print(F(E(expr)))
    expr = parse(r'(\f. \y. f y) (\x. y) q')
    print(F(E(expr)))
    expr = parse(r'(\a. a) (\x. x) (\y. y)')
    print(F(E(expr)))
    expr = parse(r'(\hi. hi)')
    print(F(E(expr)))
    expr = parse(r'(((\ x. (\ y. x)) (\ a. a)) ((\x. (x x)) (\x. (x x))))')
    print(F(E(expr)))
