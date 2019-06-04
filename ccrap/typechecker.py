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

def runseq(inp, stack, seq):
    for tok, val in seq:
        if tok == 'int':
            stack.append(('int', val))
        elif tok == 'quot':
            stack.append(('quot', val))
        elif val == 'call':
            if not stack:
                err = 'Cannot infer call!'
                raise TypeCheckError(err)
            tok2, val2 = stack.pop()
            if tok2 != 'quot':
                err = 'Call needs literal quotation!'
                raise TypeCheckError(err)
            inp, stack = runseq(inp, stack, val2)
        elif val == 'dip':
            tok2, val2 = stack.pop()
            if tok2 != 'quot':
                err = 'Call needs literal quotation!'
                raise TypeCheckError(err)
            ensure(inp, stack, 1)
            saved = stack.pop()
            inp, stack = runseq(inp, stack, val2)
            stack.append(saved)
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
    inp, stack = runseq([], [], seq)
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
