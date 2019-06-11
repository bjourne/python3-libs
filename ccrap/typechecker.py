# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser
from ccrap.lexer import Lexer
from ccrap.parser import Parser
from collections import defaultdict
from string import ascii_lowercase
from sys import exit

class TypeCheckError(Exception):
    def __init__(self, message):
        super().__init__(message)

NEXT_NAME = -1
def gensym():
    global NEXT_NAME
    NEXT_NAME = (NEXT_NAME + 1) % len(ascii_lowercase)
    return ascii_lowercase[NEXT_NAME]

def format_side(side):
    seq = [str(e) if not type(e) == tuple else format(e)
           for e in side]
    return ' %s ' % ' '.join(seq) if seq else ' '

def format(eff):
    ins = format_side(eff[0])
    outs = format_side(eff[1])
    return '(%s--%s)' % (ins, outs)

def parse_effect(str):
    parser = Parser(Lexer(str))
    return parser.parse_effect()

def combine(inp1, stack1, inp2, stack2):
    err = 'Cannot combine incompatible effects, %s and %s.'
    if len(stack1) - len(inp1) != len(stack2) - len(inp2):
        eff1_str = format(rename(inp1, stack1))
        eff2_str = format(rename(inp2, stack2))
        raise TypeCheckError(err % (eff1_str, eff2_str))

    # Ensure stacks are aligned
    out_len = max(len(stack1), len(stack2))
    ensure(inp1, stack1, out_len)
    ensure(inp2, stack2, out_len)
    assert len(inp1) == len(inp2)
    inp3 = inp1
    stack3 = []
    seen = {}
    for el1, el2 in zip(stack1, stack2):
        if el1 in inp1 and el2 in inp2 and inp1.index(el1) == inp2.index(el2):
            stack3.append(el1)
        elif el1 == el2:
            stack3.append(el1)
        else:
            if not el1 in seen:
                seen[el1] = 'dyn', gensym()
            stack3.append(seen[el1])
    return inp3, stack3

BUILTINS = {
    '<' : parse_effect('( a b -- c )'),
    '+' : parse_effect('( a b -- c )'),
    '-' : parse_effect('( a b -- c )'),
    '2drop' : parse_effect('( a b -- )'),
    'drop' : parse_effect('( a -- )'),
    'dup' : parse_effect('( a -- a a )'),
    'nip' : parse_effect('( a b -- b )'),
    'swap' : parse_effect('( a b -- b a )'),
    'tuck' : parse_effect('( x y -- y x y )')
}

def ensure(inp, stack, cnt):
    n = cnt - len(stack)
    if n <= 0:
        return
    syms = [('dyn', gensym()) for _ in range(n)]
    while syms:
        sym = syms.pop()
        inp.insert(0, sym)
        stack.insert(0, sym)

def apply_effect(inp, stack, eff):
    ins, outs = eff
    ensure(inp, stack, len(ins))
    n_ins = len(ins)
    new_outs = [stack[-(n_ins - ins.index(el))] if el in ins
                else ('dyn', gensym())
                for el in outs]
    for _ in ins:
        stack.pop()
    stack.extend(new_outs)

def apply_item(inp, stack, item):
    tok, val = item
    if tok == 'quot':
        return apply_quot(inp, stack, val)
    elif tok == 'either':
        item1, item2 = val
        inp1, stack1 = apply_item(list(inp), list(stack), item1)
        inp2, stack2 = apply_item(list(inp), list(stack), item2)
        return combine(inp1, stack1, inp2, stack2)
    err = 'Call and dip needs literal quotation!'
    raise TypeCheckError(err)

def apply_call(inp, stack):
    ensure(inp, stack, 1)
    item = stack.pop()
    return apply_item(inp, stack, item)

def apply_dip(inp, stack):
    ensure(inp, stack, 2)
    item = stack.pop()
    saved = stack.pop()
    inp, stack = apply_item(inp, stack, item)
    stack.append(saved)
    return inp, stack

def apply_quot(inp, stack, seq):
    for tok, val in seq:
        if tok == 'int':
            stack.append(('int', val))
        elif tok == 'quot':
            stack.append(('quot', val))
        elif val == 'call':
            inp, stack = apply_call(inp, stack)
        elif val == 'dip':
            inp, stack = apply_dip(inp, stack)
        elif val == '?':
            ensure(inp, stack, 3)
            item1 = stack.pop()
            item2 = stack.pop()
            stack.pop()
            stack.append(('either', (item1, item2)))
        else:
            apply_effect(inp, stack, BUILTINS[val])
    return inp, stack

def rename(ins, outs):
    global NEXT_NAME
    NEXT_NAME = -1
    slots = ins + outs
    conv = {}
    for tokval in slots:
        tok, val = tokval
        if tokval not in conv:
            if tok == 'int':
                conv[tokval] = val
            else:
                conv[tokval] = gensym()
    new_ins = tuple([conv[n] for n in ins])
    new_outs = tuple([conv[n] for n in outs])
    return new_ins, new_outs

def typecheck(seq):
    inp, stack = apply_quot([], [], seq)
    return rename(inp, stack)

if __name__ == '__main__':
    parser = ArgumentParser(description = 'CCrap type-checker')
    parser.add_argument('--quot', '-q',
                        type = str, required = True,
                        help = 'Quotation to type-check')
    args = parser.parse_args()
    parser = Parser(Lexer(args.quot))
    tp, val = parser.next_token()
    quot = parser.parse_token(tp, val)
    print(format(typecheck(quot[1])))
