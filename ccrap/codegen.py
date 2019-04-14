from ccrap.lexer import Lexer
from ccrap.mangler import generate_name, mangle
from ccrap.optimizer import optimize
from ccrap.parser import Parser

########################################################################
# Small set of primitives
########################################################################
PRIMITIVES = {
    '?' : ([
        'stack[top - 2] = (stack[top - 2] ? stack[top - 1] : stack[top]);',
        'top -= 2;'
    ], []),
    'call' : ([
        'top--;',
        'top = ((quot_ptr *)(stack[top + 1]))(stack, top);'
    ], []),

    # Stack shuffling
    'dip' : ([
        'top -= 2;',
        'cell $save = stack[top + 1];',
        'top = ((quot_ptr *)(stack[top + 2]))(stack, top);',
        'stack[++top] = $save;'
    ], ['$save']),
    'drop' : ([
        'top--;'
    ], []),
    'dup' : ([
        'top++;',
        'stack[top] = stack[top - 1];'
    ], []),
    'dupd' : ([
        'top++;',
        'stack[top] = stack[top - 1];',
        'stack[top - 1] = stack[top - 2];'
    ], []),
    'over' : ([
        'top++;',
        'stack[top] = stack[top - 2];'
    ], []),
    'pick' : ([
        'top++;',
        'stack[top] = stack[top - 3];'
    ], []),
    'spin' : ([
        'stack[top + 1] = stack[top];',
        'stack[top] = stack[top - 2];',
        'stack[top - 2] = stack[top + 1];'
    ], []),
    'swap' : ([
        'stack[top + 1] = stack[top];',
        'stack[top] = stack[top - 1];',
        'stack[top - 1] = stack[top + 1];'
    ], []),

    # C interop
    'deref' : ([
        'stack[top - 1] = ((cell *)stack[top - 1])[stack[top]];',
        'top--;'
    ], [])
}
ARITH_OPS = '+-*/'
for op in ARITH_OPS:
    PRIMITIVES[op] = ([
        'stack[top - 1] %s= stack[top];' % op,
        'top--;'
    ], [])

CMP_OPS = [
    ('=', '=='),
    ('>', '>'),
    ('<', '<')
]
for name, c_name in CMP_OPS:
    PRIMITIVES[name] = ([
        'stack[top - 1] = stack[top - 1] %s stack[top];' % c_name,
        'top--;'
    ], [])

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
#include <stdio.h>

typedef uintptr_t cell;
typedef int (quot_ptr)(cell stack[256], int top);

// Forward declarations for c bindings
{0}

// Forward declarations for words
{1}

// Word definitions
{2}

int
main(int argc, char* argv[]) {{
    cell stack[256] = {{0}};
    int top = -1;
    stack[0] = argc;
    stack[1] = (cell) argv;
    top += 2;
    top = {3}(stack, top);
    top++;
    if (top) {{
        printf("--- Remaining:\\n");
        for (int i = 0; i < top; i++) {{
            printf("    %-2d %lld\\n", top - i - 1, stack[i]);
        }}
    }}
    return 0;
}}'''

def emit_call(val):
    if val in PRIMITIVES:
        insns = insert_vars(*PRIMITIVES[val])
    else:
        insns = ['top = %s(stack, top);' % mangle(val)]
    return insns

def emit_literal(val):
    return ['stack[++top] = (cell)%s;' % val]

def emit_quot_literal(val):
    return ['stack[++top] = (cell)&%s;' % mangle(val)]

def emit_if(funcs, val):
    true_seq, false_seq = val

    true_insns = generate_body(funcs, true_seq)
    false_insns = generate_body(funcs, false_seq)
    block = [
        'top--;',
        'if (stack[top + 1]) {'
        ]
    block.extend('    %s' % insn for insn in true_insns)
    block.append('} else {')
    block.extend('    %s' % insn for insn in false_insns)
    block.append('}')
    return block

def insert_vars(insns, vars):
    for var in vars:
        name = generate_name('var')
        insns = [insn.replace(var, name) for insn in insns]
    return insns

def emit_for_index(funcs, val):
    block = [
        'cell $to = stack[top--];',
        'for (int $idx = 0; $idx < $to; $idx++) {',
        '    stack[++top] = $idx;'
    ]
    block = insert_vars(block, ['$idx', '$to'])
    insns = generate_body(funcs, val)
    block.extend('    %s' % insn for insn in insns)
    block.append('}')
    return block

def emit_dip(funcs, val):
    block = ['cell $save = stack[top--];']
    block.extend(generate_body(funcs, val))
    block.append('stack[++top] = $save;')
    return insert_vars(block, ['$save'])


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
        elif type == 'if':
            insns.extend(emit_if(funcs, val))
        elif type == 'for-index':
            insns.extend(emit_for_index(funcs, val))
        elif type == 'dip':
            insns.extend(emit_dip(funcs, val))
        else:
            raise Exception('Type `%s` not handled!' % type)
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
    if c_name not in c_decls:
        c_decl = '%s %s(%s);' % (ret, c_name, ', '.join(types))
        c_decls[c_name] = c_decl

    # Then calling instructions
    if types[-1] == '...':
        types = types[:-1] + ['void *'] * n_var_args
    n_args = len(types)
    locs = [n_args - 1 - i for i in range(n_args)]

    fmt = '(%s)stack[top-%d]'
    args = [fmt % (type, loc) for (type, loc) in zip(types, locs)]
    args_text = ', '.join(args)
    if ret == 'void':
        insns = [
            '%s(%s);' % (c_name, args_text),
            'top -= %d;' % n_args
        ]
    else:
        insns = ['cell $ret = (cell)%s(%s);' % (c_name, args_text)]
        n_args -= 1
        if n_args == 0:
            insns.append('stack[top] = $ret;')
        elif n_args == -1:
            insns.append('stack[++top] = $ret;')
        else:
            insns.extend([
                'top -= %d;' % n_args,
                'stack[top] = $ret;'
            ])
        insns = insert_vars(insns, ['$ret'])
    funcs[name] = insns

def format_insns(insns, vars):
    text = '\n'.join('    %s' % insn for insn in insns)
    for var in vars:
        name = generate_name('var')
        text = text.replace(var, name)
    return text


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
    word_defs_text = '\n'.join(
        FMT_WORD % (mangle(k), format_insns(insns, []))
        for (k, insns) in sorted_funcs)

    return FMT_VOCAB.format(c_decls_fwd_text,
                            word_decls_text,
                            word_defs_text,
                            mangle('main')).strip()

if __name__ == '__main__':
    text = """
C: printf0 int printf ( const char* , ... ) 0
C: printf1 int printf ( const char* , ... ) 1
C: puts int puts ( const char* ) 0

C: time time_t time ( time_t* ) 0

C: fopen FILE* fopen ( const char* , const char* ) 0
C: fclose int fclose ( FILE* ) 0
C: ftell long ftell ( FILE* ) 0
C: fseek int fseek ( FILE* , long , int ) 0
C: fread size_t fread ( void* , size_t , size_t , FILE* ) 0

C: malloc void* malloc ( size_t ) 0

C: calloc void* calloc ( size_t , size_t ) 0

C: free void free ( void* ) 0

: print-string ( str -- )
    puts drop ;

: seek-to-end ( FILE* -- )
    0 2 fseek drop ;

: seek-to-ofs ( FILE* n -- )
    0 fseek drop ;

: file-size ( FILE* -- int )
    ! save current offset
    dup ftell swap
    ! n FILE
    dup seek-to-end
    dup ftell
    ! ofs FILE len
    [ swap seek-to-ofs ] dip ;

: print-int ( i -- )
    "%lld\\n" swap printf1 drop ;

: 2drop ( x y -- )
    drop drop ;

: -rot ( x y z -- z x y )
    spin swap ;

: file-contents ( FILE* -- contents )
    dup file-size dup 1 + 1 calloc dup
    [
        spin 1 -rot fread drop
    ] dip ;

: with-file-reader ( file-name quot -- ... )
    [ "rb" fopen ] dip
    over [ call ] dip fclose drop ;

: with-memory ( mem quot -- ... )
    over [ call ] dip free ;

: fib ( n -- n' )
    dup 2 < [ ] [
        dup 1 - fib
        swap 2 - fib +
    ] ? call ;

: 2dip ( x y quot -- x y )
    swap [ dip ] dip ;

: times ( n quot -- ... )
    over 0 = [ 2drop ] [
        swap over
        [ call ] 2dip
        swap 1 - swap times
    ] if ;

: fib-iter ( n -- n' )
    dup 2 < [ ] [
        [ 0 1 ] dip [
            swap dup [ + ] dip
        ] times drop
    ] if ;

: main ( argc argv -- )
    2drop 150 fib-iter print-int ;


"""
    parser = Parser(Lexer(text))
    defs = parser.parse_defs()
    defs = optimize(defs)
    text = generate_vocab(defs)
    print(text)
