from collections import namedtuple
from re import findall

LAM, DOT, LPAREN, RPAREN, LCID, EOF = range(6)

Token = namedtuple('Token', ['type', 'value'])
Ident = namedtuple('Ident', ['id'])
Appl = namedtuple('Appl', ['lhs', 'rhs'])
Abst = namedtuple('Abst', ['id', 'body'])

def abst(t):
    t.pop(0)
    params = []
    while t[0].type != DOT:
        type, value = t.pop(0)
        params.append(value)
    t.pop(0)
    abst = Abst(params.pop(), term(t))
    while params:
        abst = Abst(params.pop(), abst)
    return abst

def term(t):
    return abst(t) if t[0].type == LAM else appl(t)

def atom(t):
    peek_type = t[0].type
    if peek_type == LPAREN:
        t.pop(0)
        trm = term(t)
        t.pop(0)
        return trm
    elif peek_type == LCID:
        return Ident(t.pop(0).value)
    elif peek_type == LAM:
        return abst(t)
    return None

def appl(t):
    lhs = atom(t)
    while 1:
        rhs = atom(t)
        if not rhs:
            return lhs
        lhs = Appl(lhs, rhs)

def parse(s):
    types = {'\\' : LAM, '.' : DOT, '(' : LPAREN, ')' : RPAREN}
    parts = findall(r'(\(|\)|\\|[a-z]\w*|\.)', s)
    #print(parts)
    toks = [Token(types.get(p, LCID), p) for p in parts]
    toks.append(Token(EOF, None))
    return term(toks)

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

def eval(a):
    try:
        return eval(step(a))
    except RuntimeError:
        return a

if __name__ == '__main__':
    expr = parse(r'((\ a. (\ b. (a (a (a b))))) (\ c. (\ d. (c (c d)))))')
    print(format(eval(expr)))
    expr = parse(r'(\f y. f y) (\x. y) q')
    print(format(eval(expr)))
    expr = parse(r'(\a. a) (\x. x) (\y. y)')
    print(format(eval(expr)))
    expr = parse(r'(\hi. hi)')
    print(format(eval(expr)))
