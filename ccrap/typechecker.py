# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser
from ccrap.lexer import Lexer
from ccrap.parser import Parser
from collections import namedtuple
from string import ascii_lowercase

Effect = namedtuple('Effect', ['ins', 'outs'])

NEXT_NAME = -1
def gensym():
    global NEXT_NAME
    NEXT_NAME = (NEXT_NAME + 1) % len(ascii_lowercase)
    return ascii_lowercase[NEXT_NAME]

def dup(eff):
    if not eff.outs:
        sym = gensym()
        return Effect(eff.ins + (sym,), (sym, sym))
    else:
        return Effect(eff.ins, eff.outs + (eff.outs[-1],))

def pop(eff):
    if not eff.outs:
        sym = gensym()
        return Effect(eff.ins + (sym,), eff.outs)
    else:
        return Effect(eff.ins, eff.outs[:-1])

def push(eff, val):
    return Effect(eff.ins, eff.outs + (val,))

def apply_tokval(eff, tok, val):
    ins, outs = eff
    if val == 'dup':
        return dup(eff)
    elif val == 'drop':
        return pop(eff)
    elif val == '-':
        return push(pop(pop(eff)), gensym())
    elif tok == 'int':
        return push(eff, val)

def format(eff):
    ins = ' '.join(str(i) for i in eff.ins)
    outs = ' '.join(str(o) for o in eff.outs)
    return '( %s -- %s )' % (ins, outs)

if __name__ == '__main__':
    parser = ArgumentParser(description = 'CCrap type-checker')
    parser.add_argument('--quot', '-q',
                        type = str, required = True,
                        help = 'Quotation to type-check')
    args = parser.parse_args()
    parser = Parser(Lexer(args.quot))
    type, val = parser.next_token()
    quot = parser.parse_token(type, val)
    eff = Effect((), ())
    for (tok, val) in quot[1]:
        eff = apply_tokval(eff, tok, val)
    print(format(eff))


    #print(parser.parse_body
