# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#
# This is a toy language for writing C code in concatenative
# style. You run it like this:
#
# python ccrap.py -f test.ccrap | gcc -O3 -o main -xc - && ./main
#
# It is not meant to be taken seriously.
#
# FAQ
# ===
#
# Why so many intrinsics?
# -----------------------
# It is an easy way to improve the code clang and gcc emits.
#
# What is missing?
# ----------------
# Comments and lots more things.
#
# Dips or something
# -----------------
# Required by while-loops?
from argparse import ArgumentParser
from codecs import open
from re import findall
from sys import exit

from ast import *

COUNTERS = {'tempvar' : 0, 'lambda' : 0}
def hit_counter(counter):
    val = COUNTERS[counter]
    COUNTERS[counter] += 1
    return val

def prefix_name(name):
    return 'ccrap_%s' % name

def generate_name(type):
    return prefix_name('%s_%s' % (type, hit_counter(type)))

########################################################################
# Formatting templates
########################################################################
FMT_CDECL = 'int %s(%s);'

FMT_FWD_DECL = 'static int %s(cell stack[256], int top);'

FMT_DEF = '''static int
%s(cell stack[256], int top) {
%s
    return top;
}'''

FMT_FILE = '''#include <stdint.h>
#include <assert.h>

typedef uintptr_t cell;
typedef int (quot_ptr)(cell stack[256], int top);

// Forward declarations for c bindings
%%s

// Forward declarations for words
%%s

// Word definitions
%%s

int
main(int argc, char* argv[]) {
    cell stack[256] = {0};
    int top = -1;
    stack[0] = argc;
    stack[1] = (cell) argv;
    top += 2;
    top = %s(stack, top);
    assert(top == -1);
    return 0;
}''' % prefix_name('main')

########################################################################
# Stack shuffling
########################################################################
def emit_swap(body):
    var = generate_name('tempvar')
    body.append(Custom([
        'cell %s = stack[top];' % var,
        'stack[top] = stack[top - 1];',
        'stack[top - 1] = %s;' % var
    ]))
    return True

def emit_swapd(body):
    var = generate_name('tempvar')
    body.append(Custom([
        'cell %s = stack[top - 1];' % var,
        'stack[top - 1] = stack[top - 2];',
        'stack[top - 2] = %s;' % var
    ]))
    return True


def emit_drop(body):
    body.append(Custom(['top--;']))
    return True

def emit_dup(body):
    body.append(Custom([
        'stack[top + 1] = stack[top];',
        'top++;'
        ]))
    return True

def emit_dupd(body):
    body.append(Custom([
        'stack[top + 1] = stack[top];',
        'stack[top] = stack[top - 1];',
        'top++;'
        ]))
    return True

def emit_tuck(body):
    body.append(Custom([
        'stack[top + 1] = stack[top];',
        'stack[top] = stack[top - 1];',
        'stack[top - 1] = stack[top + 1];',
        'top++;'
        ]))
    return True

def emit_2dup(body):
    body.append(Custom([
        'stack[top + 1] = stack[top - 1];',
        'stack[top + 2] = stack[top];',
        'top += 2;'
        ]))
    return True

def emit_over(body):
    body.append(Custom([
        'stack[top + 1] = stack[top - 1];',
        'top++;'
        ]))
    return True

def emit_pick(body):
    body.append(Custom([
        'stack[top + 1] = stack[top - 2];',
        'top++;'
        ]))
    return True

def emit_spin(body):
    var = generate_name('tempvar')
    body.append(Custom([
        'cell %s = stack[top];' % var,
        'stack[top] = stack[top - 2];',
        'stack[top - 2] = %s;' % var
        ]))
    return True

########################################################################
# C interop
########################################################################
def emit_deref():
    return [
        'stack[top - 1] = ((cell *)stack[top - 1])[stack[top]];',
        'top--;'
    ]

########################################################################
# Conditions & calls
########################################################################
def emit_call(body):
    body.append(Custom([
        'top--;',
        'top = ((quot_ptr *)(stack[top + 1]))(stack, top);'
    ]))
    return True

def emit_cond(body):
    body.append(Custom([
        'stack[top - 2] = stack[top - 2] ? stack[top - 1] : stack[top];',
        'top -= 2;'
    ]))
    return True

########################################################################
# Arithmetic intrinsics
########################################################################
def emit_arith(body, ch):
    body.append(Custom([
        'stack[top - 1] %s= stack[top];' % ch,
        'top--;'
    ]))
    return True

########################################################################
# Comparisons
########################################################################
def emit_cmp(body, ch):
    body.append(Custom([
        'stack[top - 1] = stack[top - 1] %s stack[top];' % ch,
        'top--;'
        ]))
    return True

########################################################################
# Auxilliary intrinsics for optimization
########################################################################
def emit_if(body):
    if not isinstance(body[-1], Definition) or \
       not isinstance(body[-2], Definition):
        return False
    false = body.pop()
    true = body.pop()
    block = [
        'top--;',
        'if (stack[top + 1]) {'
    ]
    block += true.emit_body()
    block += ['} else {']
    block += false.emit_body()
    block += ['}']
    body.append(Custom(block))
    return True

def emit_times(body):
    if not isinstance(body[-1], Definition) or \
       not isinstance(body[-2], IntLiteral):
        return False
    quot = body.pop()
    n = body.pop()
    var = generate_name('tempvar')
    for_tmpl = 'for (int %s = 0; %s < %s; %s++) {'
    block = [for_tmpl % (var, var, n.lit, var)]
    block += quot.emit_body()
    block += ['}']
    body.append(Custom(block))
    return True

def emit_while(body):
    if not isinstance(body[-1], Definition) or \
       not isinstance(body[-2], Definition):
        return False
    loop = body.pop()
    test = body.pop()
    block = ['while (1) {']
    block += test.emit_body()
    block += [
        '    top--;',
        '    if (!stack[top + 1]) {',
        '        top--;',
        '        break;',
        '    }'
    ]
    block += loop.emit_body()
    block += ['}']
    body.append(Custom(block))
    return True

intrinsics = {
    # Conditions & calls
    'call' : emit_call,
    '?' : emit_cond,


    # Comparisons
    '=' : lambda b: emit_cmp(b, '=='),
    '!=' : lambda b: emit_cmp(b, '!='),
    '>' : lambda b: emit_cmp(b, '>'),

    # Arithmetic
    '*' : lambda b: emit_arith(b, '*'),
    '+' : lambda b: emit_arith(b, '+'),
    '-' : lambda b: emit_arith(b, '-'),

    # C-compat
    'deref' : emit_deref,

    # Stack shuffling
    '2dup' : emit_2dup,
    'dup' : emit_dup,
    'dupd' : emit_dupd,
    'drop' : emit_drop,
    'over' : emit_over,
    'pick' : emit_pick,
    'spin' : emit_spin,
    'swap' : emit_swap,
    'swapd' : emit_swapd,
    'tuck' : emit_tuck,

    # Auxilliary intrinsics for optimization
    #'if' : emit_if,
    'times' : emit_times
    #'while' : emit_while
    }

########################################################################
# Parsing stuff
########################################################################

# Globals work fine for now

# All parsed function definitions
funcs = {}

# Declarations from c calls
c_decls = set()

# Token buffer
toks = []

def parse_c_call():
    return_type = toks.pop(0)
    name = toks.pop(0)
    assert toks.pop(0) == '('
    types = []
    while True:
        t = []
        while toks[0] not in (',', ')'):
            t.append(toks.pop(0))
        types.append(' '.join(t))
        if toks.pop(0) == ')':
            break

    c_decl = FMT_CDECL % (name, ', '.join(types))
    c_decls.add(c_decl)

    n_params = toks.pop(0)

    assert toks.pop(0) == '}'
    if types[-1] == '...':
        n_params = int(n_params)
        types = types[:-1] + ['void *'] * n_params
    return CCall(name, types)

def parse_quotation():
    body = parse_body(']')
    name = generate_name('lambda')
    funcs[name] = Definition(name, body)
    return funcs[name]

def parse_body(end):
    body = []
    while True:
        tok = toks.pop(0)
        if tok == end:
            return body
        elif tok == '{':
            body.append(parse_c_call())
        elif tok == '[':
            body.append(parse_quotation())
        elif tok[0] == '"':
            body.append(StringLiteral(tok))
        elif tok.isdigit():
            body.append(IntLiteral(int(tok)))
        else:
            handled = False
            if tok in intrinsics:
                handled = intrinsics[tok](body)
            if not handled:
                body.append(Call(prefix_name(tok)))

def parse_def():
    name = prefix_name(toks.pop(0))
    body = parse_body(';')
    funcs[name] = Definition(name, body)

def parse():
    while toks:
        tok = toks.pop(0)
        if tok == ':':
            parse_def()

    sorted_funcs = list(sorted(funcs.items()))
    decls = '\n'.join(FMT_FWD_DECL % n for (n, b) in sorted_funcs)

    defs = [FMT_DEF % (f.name, '\n'.join(f.emit_body()))
            for (n, f) in sorted_funcs]
    defs_text = '\n\n'.join(defs)
    print(FMT_FILE % ('\n'.join(c_decls), decls, defs_text))

def main():
    global toks
    parser = ArgumentParser(description = 'CCrap compiler')
    parser.add_argument('--file', '-f',
                        type = str, required = True,
                        help = 'File to compile')
    args = parser.parse_args()
    with open(args.file, 'r', 'utf-8') as f:
        text = f.read()
    toks = findall("(?:\".*?\"|\S)+", text)
    parse()

if __name__ == '__main__':
    main()
