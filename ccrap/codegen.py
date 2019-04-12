from ccrap.lexer import Lexer
from ccrap.parser import Parser
from sys import exit

########################################################################
# Name mangling
########################################################################
def mangle(name):
    return 'ccrap_%s' % name

########################################################################

FMT_WORD = '''static int
%s(cell stack[256], int top) {
%s
    return top;
}'''

FMT_FWD_DECL_DEF = '''
static int %s(cell stack[256], int top);
'''

FMT_VOCAB = '''
#include <stdint.h>
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
}'''

def emit_call(val):
    return ['top = %s(stack, top);' % mangle(val)]

def emit_literal(val):
    return ['top++;', 'stack[top] = (cell)%s;' % val]

def generate_def(dfn):
    name = dfn[0]
    body = dfn[2][1]
    insns = []
    for type, val in body:
        if type == 'sym':
            insns.extend(emit_call(val))
        elif type == 'str':
            insns.extend(emit_literal(val))
    return name, insns

def generate_ccall(dfn, c_decls):
    name = dfn[0]
    ret = dfn[1]
    c_name = dfn[2]
    types = dfn[3][1]
    n_var_args = dfn[4]
    if types[-1] == '...':
        types = types[:-1] + ['void *'] * n_var_args
    n_args = len(types)
    locs = [n_args - 1 - i for i in range(n_args)]

    fmt = '(%s)stack[top - %d]'
    args = [fmt % (type, loc) for (type, loc) in zip(types, locs)]

    call = '%s(%s);' % (c_name, ', '.join(args))
    drop = 'top -= %d;' % n_args

    # Now generate text for the c_decl
    c_decl = '%s %s(%s);' % (ret, c_name, ', '.join(types))
    c_decls[c_name] = c_decl

    return name, [call, drop]

def generate_vocab(defs):
    funcs = {}
    c_decls = {}
    for type, df in defs:
        if type == 'def':
            name, insns = generate_def(df)
            funcs[name] = insns
        elif type == 'cdef':
            name, insns = generate_ccall(df, c_decls)
            funcs[name] = insns

    # C declarations text
    c_decls_fwd_text = '\n'.join(v for (k, v) in sorted(c_decls.items()))

    # Forward declarations for words
    sorted_funcs = sorted(funcs.items())
    word_decls_text = '\n'.join(
        FMT_FWD_DECL_DEF.strip() % mangle(k)
        for (k, v) in sorted_funcs)

    # Word definitions
    word_defs_text = '\n'.join(
        FMT_WORD % (mangle(k), '\n'.join(
            '    %s' % line for line in v
        ))
        for (k, v) in sorted_funcs)

    vocab_fmt = FMT_VOCAB.strip() % mangle('main')

    return vocab_fmt % (c_decls_fwd_text,
                        word_decls_text,
                        word_defs_text)

if __name__ == '__main__':
    text = """
C: printf0 void printf ( const char* , ... ) 0

: main ( argc argv -- ) "hello, world\\n" printf0 ;
"""
    parser = Parser(Lexer(text))
    defs = parser.parse_defs()
    text = generate_vocab(defs)
    print(text)
