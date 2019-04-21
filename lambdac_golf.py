from collections import *
import re
A,B,y,c=namedtuple('A',['l','r']),namedtuple('B',['i','b']),type,list.pop
def ab(t):
    c(t,0);p = c(t,0);c(t,0)
    return B(p,tm(t))
def tm(t):
    return ab(t)if t[0] == '\\'else ap(t)
def at(t):
    if t[0] == '(':
        c(t,0)
        trm = tm(t)
        c(t,0)
        return trm
    if ord('a')<=ord(t[0][0])<=ord('z'):return c(t,0)
    if t[0]=='\\':return ab(t)
def ap(t):
    l = at(t)
    while 1:
        r = at(t)
        if not r:return l
        l = A(l, r)
def P(s):
    return tm(re.findall(r'(\(|\)|\\|[a-z]\w*|\.)',s)+['='])
def V(e):o=y(e);return V(e.b)-{e.i} if o==B else V(e.l)|V(e.r)if o==A else{e}
def R(e,f,t):return B(e.i,R(e.b,f,t)) if y(e)==B else A(R(e.l,f,t),R(e.r,f,t))if y(e)==A else t if e==f else e
def N(i,e):return N(chr(97+(ord(i[0])-96)%26),e) if i in V(e)else i
def S(i,e,a):
    o=y(e)
    if o==B:return e if e.i==i else o(N(e.i,a),S(i,R(e.b,e.i,N(e.i,a)),a))
    if o==A:return o(S(i,e.l,a),S(i,e.r,a))
    return a if e==i else e
def T(e):
    o=y(e)
    if o==A:
        l,r=e
        return S(l.i,l.b,r)if y(l)==B else A(T(l),r)if y(l)==A else o(l,T(r))
    if o==B:
        return o(e.i, T(e.b))
    raise RuntimeError('hi')
def E(a):
    try: return E(T(a))
    except RuntimeError:
        return a
def F(e):
    o=y(e)
    if o==B:return r'(\%s. %s)'%(e.i,F(e.b))
    if o==A:return r'(%s %s)'%(F(e.l),F(e.r))
    return e
if __name__ == '__main__':
    expr = P(r'((\a. (\b. (a (a (a b))))) (\ c. (\ d. (c (c d)))))')
    print(F(E(expr)))
    expr = P(r'(\f. \y. f y) (\x. y) q')
    print(F(E(expr)))
    expr = P(r'(\a. a) (\x. x) (\y. y)')
    print(F(E(expr)))
    expr = P(r'(\hi. hi)')
    print(F(E(expr)))
    expr = P(r'(((\ x. (\ y. x)) (\ a. a)) ((\x. (x x)) (\x. (x x))))')
    print(F(E(expr)))
