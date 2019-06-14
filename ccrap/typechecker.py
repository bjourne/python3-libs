# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
from argparse import ArgumentParser
from ccrap.lexer import Lexer
from ccrap.parser import Parser
from collections import defaultdict, namedtuple
from string import ascii_lowercase
from sys import exit

########################################################################
# Name generation
########################################################################
NEXT_NAME = -1
def gensym():
    global NEXT_NAME
    NEXT_NAME = (NEXT_NAME + 1) % len(ascii_lowercase)
    return ascii_lowercase[NEXT_NAME]

########################################################################
# Keeping track of the stack state
########################################################################
StackState = namedtuple('StackState', ['ins', 'outs'])

def clone(state):
    return StackState(list(state.ins), list(state.outs))

def height(state):
    return len(state.outs) - len(state.ins)

def compatible_items(state1, el1, state2, el2):
    if ((el1 in state1.ins and el2 in state2.ins and
         state1.ins.index(el1) == state2.ins.index(el2)) or
        el1 == el2):
        return True
    return False

def rename_rec(eff, conv):
    ins, outs = eff
    for tokval in ins + outs:
        tok, val = tokval
        if tokval not in conv:
            if tok == 'int':
                conv[tokval] = val
            elif tok == 'effect':
                conv[tokval] = rename_rec(val, conv)
            else:
                conv[tokval] = gensym()
    new_ins = tuple([conv[n] for n in ins])
    new_outs = tuple([conv[n] for n in outs])
    return new_ins, new_outs

def rename(eff):
    global NEXT_NAME
    NEXT_NAME = -1
    return rename_rec(eff, {})

########################################################################

class TypeCheckError(Exception):
    def __init__(self, message):
        super().__init__(message)

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
    out_len = max(len(state1.outs), len(state2.outs))
    ensure(state1, out_len)
    ensure(state2, out_len)
    assert len(state1.ins) == len(state2.ins)

    state3 = StackState(state1.ins, [])
    seen = {}
    for el1, el2 in zip(state1.outs, state2.outs):
        if compatible_items(state1, el1, state2, el2):
            state3.outs.append(el1)
        else:
            if not el1 in seen:
                seen[el1] = 'dyn', gensym()
            state3.outs.append(seen[el1])
    return state3

BUILTINS = {
    '<' : parse_effect('( a b -- c )'),
    '+' : parse_effect('( a b -- c )'),
    '*' : parse_effect('( a b -- c )'),
    '-' : parse_effect('( a b -- c )'),
    '=' : parse_effect('( a b -- c )'),
    '2drop' : parse_effect('( a b -- )'),
    '2dup' : parse_effect('( a b -- a b a b )'),
    'drop' : parse_effect('( a -- )'),
    'dup' : parse_effect('( a -- a a )'),
    'nip' : parse_effect('( a b -- b )'),
    'over' : parse_effect('( a b -- a b a )'),
    'swap' : parse_effect('( a b -- b a )'),
    'tuck' : parse_effect('( x y -- y x y )'),

    'foo' : parse_effect('( -- ( -- ) )'),
    'foo1' : parse_effect('( -- ( -- a ) )'),
    'foo-int' : parse_effect('( -- ( -- 10 20 ) )')
}

def ensure(state, cnt):
    for _ in range(cnt - len(state.outs)):
        sym = 'dyn', gensym()
        state.ins.insert(0, sym)
        state.outs.insert(0, sym)

def apply_effect(state, eff):
    ins, outs = eff
    ensure(state, len(ins))
    n_ins = len(ins)

    new_outs = []
    seen = {}
    for tokval in outs:
        tok, val = tokval
        if tokval in ins:
            item = state.outs[-(n_ins - ins.index(tokval))]
        elif tok in ('int', 'effect'):
            item = tokval
        else:
            if tokval not in seen:
                seen[tokval] = 'dyn', gensym()
            item = seen[tokval]
        new_outs.append(item)
    for _ in ins:
        state.outs.pop()
    state.outs.extend(new_outs)
    return state

def apply_item(state, item):
    if item[0] == 'quot':
        return apply_quot(state, item)
    elif item[0] == 'either':
        item1, item2 = item[1]
        state1 = apply_item(clone(state), item1)
        state2 = apply_item(clone(state), item2)
        return combine(state1, state2)
    elif item[0] == 'effect':
        return apply_effect(state, item[1])
    err = 'Call and dip needs literal quotations!'
    raise TypeCheckError(err)

def apply_call(state):
    ensure(state, 1)
    item = state.outs.pop()
    return apply_item(state, item)

def apply_dip(state):
    ensure(state, 2)
    item = state.outs.pop()
    saved = state.outs.pop()
    state = apply_item(state, item)
    state.outs.append(saved)
    return state

def apply_qm(state):
    ensure(state, 3)
    item1 = state.outs.pop()
    item2 = state.outs.pop()
    state.outs.pop()
    state.outs.append(('either', (item1, item2)))

def apply_compose(state):
    ensure(state, 2)
    tok2, val2 = state.outs.pop()
    tok1, val1 = state.outs.pop()
    assert tok1 == 'quot' and tok2 == 'quot'
    state.outs.append(('quot', val1 + val2))

def apply_quot(state, quot):
    for tok, val in quot[1]:
        if tok == 'int':
            state.outs.append(('int', val))
        elif tok == 'quot':
            state.outs.append(('quot', val))
        elif val == 'call':
            state = apply_call(state)
        elif val == 'dip':
            state = apply_dip(state)
        elif val == 'compose':
            apply_compose(state)
        elif val == '?':
            apply_qm(state)
        else:
            apply_effect(state, BUILTINS[val])
    return state

def infer_item(item):
    if item[0] == 'quot':
        return 'effect', infer_quot(item)
    elif item[0] == 'either':
        print(item)
        item1, item2 = item[1]
        state1 = apply_item(StackState([], []), item1)
        state2 = apply_item(StackState([], []), item2)
        state3 = combine(state1, state2)
        return 'effect', (tuple(state3.ins), tuple(state3.outs))
    else:
        return item

def infer_quot(quot):
    state = StackState([], [])
    state = apply_quot(state, quot)
    outs = map(infer_item, state.outs)
    return tuple(state.ins), tuple(outs)

def infer(quot):
    state = infer_quot(quot)
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

    state = infer(quot)
    print(format(state))
