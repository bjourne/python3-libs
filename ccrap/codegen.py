from ccrap.lexer import Lexer
from ccrap.mangler import generate_name, mangle
from ccrap.parser import Parser

########################################################################
# Small set of primitives
########################################################################
PRIMITIVES = [
    ('?', [
        'stack[top - 2] = (stack[top - 2] ? stack[top - 1] : stack[top]);',
        'top -= 2;'
    ]),
    ('call', [
        'top--;',
        'top = ((quot_ptr *)(stack[top + 1]))(stack, top);'
    ]),
    # Stack shuffling
    ('drop', ['top--;']),
    ('dip', [
        'top -= 2;',
        'cell dip_save = stack[top + 1];',
        'top = ((quot_ptr *)(stack[top + 2]))(stack, top);',
        'stack[++top] = dip_save;'
    ]),
    ('dup', [
        'top++;',
        'stack[top] = stack[top - 1];'
    ]),
    ('over', [
        'top++;',
        'stack[top] = stack[top - 2];'
    ]),
    ('swap', [
        'stack[top + 1] = stack[top];',
        'stack[top] = stack[top - 1];',
        'stack[top - 1] = stack[top + 1];'
    ])
]
ARITH_OPS = '+-*/'
for op in ARITH_OPS:
    PRIMITIVES.append(
        (op, [
            'stack[top - 1] %s= stack[top];' % op,
            'top--;'
        ])
    )

CMP_OPS = [
    ('=', '=='),
    ('>', '>'),
    ('<', '<')
]
for name, c_name in CMP_OPS:
    PRIMITIVES.append(
        (name, [
            'stack[top - 1] = stack[top - 1] %s stack[top];' % c_name,
            'top--;'
        ])
    )

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

// Primitive words
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
    return ['stack[++top] = (cell)%s;' % val]

def emit_quot_literal(val):
    return ['stack[++top] = (cell)&%s;' % mangle(val)]

def generate_body(funcs, body):
    insns = []
    for type, val in body:
        if type == 'sym':
            insns.extend(emit_call(val))
        elif type == 'str':
            insns.extend(emit_literal(val))
        elif type == 'int':
            insns.extend(emit_literal(val))
        elif type == 'quot':
            quot_name = generate_quot(funcs, val)
            insns.extend(emit_quot_literal(quot_name))
    return insns

def generate_quot(funcs, body):
    name = generate_name('quot')
    funcs[name] = generate_body(funcs, body)
    return name

def generate_def(funcs, dfn):
    name = dfn[0]
    body = dfn[2]
    funcs[name] = generate_body(funcs, body)

def generate_ccall(funcs, dfn, c_decls):
    name = dfn[0]
    ret = dfn[1]
    c_name = dfn[2]
    types = dfn[3][1]
    n_var_args = dfn[4]

    # First generate text for the c_decl
    c_decl = '%s %s(%s);' % (ret, c_name, ', '.join(types))
    c_decls[c_name] = c_decl

    # Then calling instructions
    if types[-1] == '...':
        types = types[:-1] + ['void *'] * n_var_args
    n_args = len(types)
    locs = [n_args - 1 - i for i in range(n_args)]

    fmt = '(%s)stack[top - %d]'
    args = [fmt % (type, loc) for (type, loc) in zip(types, locs)]

    call = '%s(%s);' % (c_name, ', '.join(args))
    drop = 'top -= %d;' % n_args

    funcs[name] = [call, drop]

def format_insns(insns):
    return '\n'.join('    %s' % insn for insn in insns)

def generate_vocab(defs):
    funcs = {}
    c_decls = {}
    for type, df in defs:
        if type == 'def':
            generate_def(funcs, df)
        elif type == 'cdef':
            generate_ccall(funcs, df, c_decls)

    # C declarations text
    c_decls_fwd_text = '\n'.join(v for (k, v) in sorted(c_decls.items()))

    # Forward declarations for words
    sorted_funcs = sorted(funcs.items())
    word_decls_text = '\n'.join(
        FMT_FWD_DECL_DEF.strip() % mangle(k)
        for (k, v) in sorted_funcs)

    # Word definitions
    word_defs_text = '\n'.join(FMT_WORD % (mangle(k), format_insns(v))
                               for (k, v) in sorted_funcs)

    # Primitives
    primitives = [
        ('drop', ['top--;'])
        ]
    primitives = [FMT_WORD % (mangle(name), format_insns(insns))
                  for (name, insns) in PRIMITIVES]
    primitives_text = '\n'.join(primitives)

    vocab_fmt = FMT_VOCAB.strip() % mangle('main')

    return vocab_fmt % (c_decls_fwd_text,
                        word_decls_text,
                        primitives_text,
                        word_defs_text)

if __name__ == '__main__':
    text = """
C: printf0 int printf ( const char* , ... ) 0

C: printf1 int printf ( const char* , ... ) 1

C: puts int puts ( const char* ) 0

: print-int ( n -- )
    "%d\\n" swap printf1 ;

: 2drop ( x y -- )
    drop drop ;

: if ( bool quot quot -- ... )
    ? call ;

: times ( n quot -- ... )
    over 0 =
    [ 2drop ] [
        swap over
        [ call ] 2dip
        swap 1 - swap times
    ] if ;

: 2dip ( x y quot -- x y )
    swap [ dip ] dip ;

: main ( argc argv -- )
    2drop 20 20 [ 1 - dup print-int ] times drop ;
"""
    parser = Parser(Lexer(text))
    defs = parser.parse_defs()
    text = generate_vocab(defs)
    print(text)
