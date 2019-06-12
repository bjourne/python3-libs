# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser
from ccrap.lexer import Lexer
from ccrap.parser import Parser
from collections import defaultdict, namedtuple
from string import ascii_lowercase
from sys import exit

########################################################################
# Keeping track of the stack state
########################################################################
StackState = namedtuple('StackState', ['input', 'stack'])

def clone(state):
    return StackState(list(state.input), list(state.stack))

def height(state):
    return len(state.stack) - len(state.input)

def compatible_items(state1, el1, state2, el2):
    if ((el1 in state1.input and el2 in state2.input and
         state1.input.index(el1) == state2.input.index(el2)) or
        el1 == el2):
        return True
    return False

########################################################################

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

def combine(state1, state2):
    err = 'Cannot combine incompatible effects, %s and %s.'
    if height(state1) != height(state2):
        eff1_str = format(rename(state1))
        eff2_str = format(rename(state2))
        raise TypeCheckError(err % (eff1_str, eff2_str))

    # Ensure stacks are aligned
    out_len = max(len(state1.stack), len(state2.stack))
    ensure(state1, out_len)
    ensure(state2, out_len)
    assert len(state1.input) == len(state2.input)

    state3 = StackState(state1.input, [])
    seen = {}
    for el1, el2 in zip(state1.stack, state2.stack):
        if compatible_items(state1, el1, state2, el2):
            state3.stack.append(el1)
        else:
            if not el1 in seen:
                seen[el1] = 'dyn', gensym()
            state3.stack.append(seen[el1])
    return state3

BUILTINS = {
    '<' : parse_effect('( a b -- c )'),
    '+' : parse_effect('( a b -- c )'),
    '-' : parse_effect('( a b -- c )'),
    '2drop' : parse_effect('( a b -- )'),
    '2dup' : parse_effect('( a b -- a b a b )'),
    'drop' : parse_effect('( a -- )'),
    'dup' : parse_effect('( a -- a a )'),
    'nip' : parse_effect('( a b -- b )'),
    'over' : parse_effect('( a b -- a b a )'),
    'swap' : parse_effect('( a b -- b a )'),
    'tuck' : parse_effect('( x y -- y x y )')
}

def ensure(state, cnt):
    for _ in range(cnt - len(state.stack)):
        sym = 'dyn', gensym()
        state.input.insert(0, sym)
        state.stack.insert(0, sym)

def apply_effect(state, eff):
    ins, outs = eff
    ensure(state, len(ins))
    n_ins = len(ins)
    new_outs = [state.stack[-(n_ins - ins.index(el))] if el in ins
                else ('dyn', gensym())
                for el in outs]
    for _ in ins:
        state.stack.pop()
    state.stack.extend(new_outs)

def apply_item(state, item):
    tok, val = item
    if tok == 'quot':
        return apply_quot(state, val)
    elif tok == 'either':
        item1, item2 = val
        state1 = apply_item(clone(state), item1)
        state2 = apply_item(clone(state), item2)
        return combine(state1, state2)
    err = 'Call and dip needs literal quotations!'
    raise TypeCheckError(err)

def apply_call(state):
    ensure(state, 1)
    item = state.stack.pop()
    return apply_item(state, item)

def apply_dip(state):
    ensure(state, 2)
    item = state.stack.pop()
    saved = state.stack.pop()
    state = apply_item(state, item)
    state.stack.append(saved)
    return state

def apply_qm(state):
    ensure(state, 3)
    item1 = state.stack.pop()
    item2 = state.stack.pop()
    state.stack.pop()
    state.stack.append(('either', (item1, item2)))

def apply_quot(state, seq):
    for tok, val in seq:
        if tok == 'int':
            state.stack.append(('int', val))
        elif tok == 'quot':
            state.stack.append(('quot', val))
        elif val == 'call':
            state = apply_call(state)
        elif val == 'dip':
            state = apply_dip(state)
        elif val == '?':
            apply_qm(state)
        else:
            apply_effect(state, BUILTINS[val])
    return state

def rename(state):
    global NEXT_NAME
    NEXT_NAME = -1

    slots = state.input + state.stack
    conv = {}
    for tokval in slots:
        tok, val = tokval
        if tokval not in conv:
            if tok == 'int':
                conv[tokval] = val
            else:
                conv[tokval] = gensym()
    new_ins = tuple([conv[n] for n in state.input])
    new_outs = tuple([conv[n] for n in state.stack])
    return new_ins, new_outs

def infer(seq):
    state = StackState([], [])
    state = apply_quot(state, seq)
    print(state)
    return rename(state)

if __name__ == '__main__':
    parser = ArgumentParser(description = 'CCrap type-checker')
    parser.add_argument('--quot', '-q',
                        type = str, required = True,
                        help = 'Quotation to type-check')
    args = parser.parse_args()
    parser = Parser(Lexer(args.quot))
    tp, val = parser.next_token()
    quot = parser.parse_token(tp, val)
    print(format(infer(quot[1])))
