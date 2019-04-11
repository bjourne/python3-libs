# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#
# This is a toy language for writing C code in concatenative
# style. You run it like this:
#
# python ccrap.py -f test.ccrap | gcc -O3 -o main -xc - && ./main
#
# It is not meant to be taken seriously.
from argparse import ArgumentParser
from codecs import open
from re import findall

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

def push_literal(tok):
    return [
        'top++;',
        'stack[top] = (cell)%s;' % tok
    ]

def push_call(tok):
    return ['top = %s(stack, top);' % prefix_name(tok)]

########################################################################
# Stack shuffling
########################################################################
def emit_swap():
    var = generate_name('tempvar')
    return [
        'cell %s = stack[top];' % var,
        'stack[top] = stack[top - 1];',
        'stack[top - 1] = %s;' % var
    ]

def emit_drop():
    return ['top--;']

def emit_dup():
    return [
        'stack[top + 1] = stack[top];',
        'top++;'
        ]

def emit_dupd():
    return [
        'stack[top + 1] = stack[top];',
        'stack[top] = stack[top - 1];',
        'top++;'
        ]

def emit_tuck():
    return [
        'stack[top + 1] = stack[top];',
        'stack[top] = stack[top - 1];',
        'stack[top - 1] = stack[top + 1];',
        'top++;'
        ]

def emit_2dup():
    return [
        'stack[top + 1] = stack[top - 1];',
        'stack[top + 2] = stack[top];',
        'top += 2;'
        ]

def emit_over():
    return [
        'stack[top + 1] = stack[top - 1];',
        'top++;'
        ]

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
def emit_call():
    return [
        'top--;',
        'top = ((quot_ptr *)(stack[top + 1]))(stack, top);'
        ]

def emit_cond():
    return [
        'stack[top - 2] = stack[top - 2] ? stack[top - 1] : stack[top];',
        'top -= 2;'
    ]

########################################################################
# Arithmetic intrinsics
########################################################################
def emit_arith(ch):
    return [
        'stack[top - 1] %s= stack[top];' % ch,
        'top--;'
    ]

########################################################################
# Comparisons
########################################################################
def emit_cmp(ch):
    return [
        'stack[top - 1] = stack[top - 1] %s stack[top];' % ch,
        'top--;'
        ]

intrinsics = {
    # Conditions & calls
    'call' : emit_call,
    '?' : emit_cond,

    # Comparisons
    '=' : lambda: emit_cmp('=='),
    '>' : lambda: emit_cmp('>'),

    # Arithmetic
    '+' : lambda: emit_arith('+'),
    '-' : lambda: emit_arith('-'),

    # C-compat
    'deref' : emit_deref,

    # Stack shuffling
    '2dup' : emit_2dup,
    'dup' : emit_dup,
    'dupd' : emit_dupd,
    'drop' : emit_drop,
    'over' : emit_over,
    'swap' : emit_swap,
    'tuck' : emit_tuck
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

def parse_effect():
    inputs = []
    while True:
        tok = toks.pop(0)
        if tok == '--':
            break
        inputs.append(tok)
    outputs = []
    while True:
        tok = toks.pop(0)
        if tok == ')':
            break
        outputs.append(tok)
    return (inputs, outputs)

def parse_c_call():
    assert toks.pop(0) == '('
    inputs, outputs = parse_effect()
    name = toks.pop(0)
    assert toks.pop(0) == '}'

    n_args = len(inputs)

    c_decl = FMT_CDECL % (name, ', '.join(inputs))
    c_decls.add(c_decl)

    # Format args
    fmt_ref_stack = '(%s)stack[top - %d]'
    inputs = [(n_args - 1 - i, t) for (i, t) in enumerate(inputs)]
    inputs = [fmt_ref_stack % (t, i) for (i, t) in inputs]
    return ['%s(%s); top -= %d;' % (name, ', '.join(inputs), n_args)]

def parse_quotation():
    body = parse_body(']')
    name = generate_name('lambda')
    funcs[name] = body
    return [
        'top++;',
        'stack[top] = (cell)&%s;' % name
        ]

def parse_body(end):
    body = []
    while True:
        tok = toks.pop(0)
        if tok == end:
            return body
        elif tok == '{':
            body.extend(parse_c_call())
        elif tok == '[':
            body.extend(parse_quotation())
        elif tok[0] == '"' or tok.isdigit():
            body.extend(push_literal(tok))
        elif tok in intrinsics:
            body.extend(intrinsics[tok]())
        else:
            body.extend(push_call(tok))

def parse_def():
    name = prefix_name(toks.pop(0))
    funcs[name] = parse_body(';')

def parse():
    while toks:
        tok = toks.pop(0)
        if tok == ':':
            parse_def()

    decls = '\n'.join(FMT_FWD_DECL % n for n in funcs.keys())
    defs = '\n\n'.join(FMT_DEF % (n, '\n    '.join(b))
                       for (n, b) in funcs.items())
    print(FMT_FILE % ('\n'.join(c_decls), decls, defs))

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
