from ccrap.lexer import Lexer, LexerError
from ccrap.parser import Parser
from ccrap.typechecker import (BUILTINS,
                               StackState,
                               apply_effect,
                               combine, format, infer, parse_effect,
                               rename)

def test_combine():
    # Corresponds to dup
    state1 = StackState([('dyn', 'a')], [('dyn', 'a'), ('dyn', 'a')])
    state2 = StackState([], [('int', 50)])
    state3 = combine(state1, state2)
    assert state3.ins == [('dyn', 'a')]
    assert state3.outs == [('dyn', 'a'), ('dyn', 'b')]

def sym(x):
    return 'sym', x

def effect(l, r):
    return 'effect', (l, r)

def test_parse_effect():
    examples = [
        ('( a -- b )', ((sym('a'),), (sym('b'),))),
        ('( a ( b -- c ) -- a )', ((sym('a'), effect((sym('b'),), (sym('c'),))), (sym('a'),))),
        ('( -- 10 )', ((), (('int', 10),)))
        ]
    for inp, expected_out in examples:
        out = parse_effect(inp)
        assert out == expected_out

def test_apply_effect():
    state = StackState([], [])
    eff = parse_effect('( -- a a )')
    apply_effect(state, eff)
    assert format(rename(state)) == '( -- a a )'

    state = StackState([], [])
    eff = parse_effect('( -- 1 2 3 )')
    apply_effect(state, eff)
    assert format(rename(state)) == '( -- 1 2 3 )'

    state = StackState([], [])
    eff = parse_effect('( -- ( -- ) )')
    apply_effect(state, eff)
    assert format(rename(state)) == '( -- ( -- ) )'

def test_typechecking():

    # For testing purposes
    BUILTINS['foo'] = parse_effect('( -- ( -- ) )')
    BUILTINS['foo1'] = parse_effect('( -- ( -- a ) )')

    examples = [
        ('[ swap ]', '( a b -- b a )'),
        ('[ swap swap ]', '( a b -- a b )'),
        ('[ + ]', '( a b -- c )'),
        ('[ 1 swap ]', '( a -- 1 a )'),
        ('[ + - ]', '( a b c -- d )'),
        ('[ nip ]', '( a b -- b )'),

        # Typechecking nested effects
        ('[ dup [ ] dup ]', '( a -- a a ( -- ) ( -- ) )'),
        ('[ [ ] ]', '( -- ( -- ) )'),
        ('[ [ [ [ ] ] ] ]', '( -- ( -- ( -- ( -- ) ) ) )'),

        # Calling
        ('[ [ ] call ]', '( -- )'),
        ('[ [ 3 ] call ]', '( -- 3 )'),
        ('[ [ 1234 ] 44 [ call ] swap drop call ]', '( -- 1234 )'),

        # Dipping
        ('[ [ 10 swap ] 4 [ call ] dip ]', '( a -- 10 a 4 )'),
        ('[ [ ] dip ]', '( a -- a )'),
        ('[ [ [ [ ] dip ] dip ] dip ]', '( a b c -- a b c )'),
        ('[ [ 3 ] [ 4 ] swap call [ call ] dip ]', '( -- 4 3 )'),

        # Either types
        ('[ [ ] [ ] ? call ]', '( a -- )'),
        ('[ [ dup ] [ dup ] ? call ]', '( a b -- a a )'),
        ('[ [ dup ] [ 77 ] ? call ]', '( a b -- a c )'),
        ('[ [ 99 ] [ 77 ] ? call ]', '( a -- b )'),
        ('[ 7 [ dup ] [ dup ] ? call ]', '( a -- a a )'),
        ('[ 7 [ 7 7 ] [ 8 dup ] ? call ]', '( -- a a )'),

        ('[ [ 4 ] [ 4 ] ? call ]', '( a -- 4 )'),
        ('[ 0 0 [ 4 ] [ 4 ] ? [ 4 ] ? call ]', '( -- 4 )'),

        # Either with hofs
        ('[ [ ] [ ] ? ]', '( a -- ( -- ) )'),
        ('[ 0 [ 10 ] [ dup ] ? ]', '( -- ( a -- a b ) )'),

        ('[  [ [ ] ]  [ [ ] ] ? ]', '( a -- ( -- ( -- ) ) )'),
        ('[ [ ] 99 ? ]', '( a -- b )'),
        ('[ [ [ 3 ] ] [ [ ] ] ? ]', '( a -- ( -- b ) )'),

        ('[ 10 foo1 [ 3 ] ? ]', '( -- ( -- a ) )'),

        # Nested either types
        ('[ 0 0 [ ] [ ] ? [ ] ? call ]', '( -- )'),

        # Dip with either types
        ('[ [ ] [ ] ? dip ]', '( a b -- a )'),

        # Calling hofs
        ('[ foo call ]', '( -- )'),
        ('[ foo1 call ]', '( -- a )'),

        # Composing
        ('[ [ ] [ ] ++ call ]', '( -- )'),
        ('[ [ 12 ] [ 24 ] ++ ]', '( -- ( -- 12 24 ) )'),
        ('[ [ 3 ] [ 5 ] ++ ]', '( -- ( -- 3 5 ) )')
    ]
    for inp, expected_out in examples:
        parser = Parser(Lexer(inp))
        quot = parser.parse_token(*parser.next_token())
        out = format(infer(quot))
        if out != expected_out:
            print(out)
        assert out == expected_out

def test_tokens():
    examples = [
        ('pooo"ee', ['pooo"ee']),
        (': hej', [':', 'hej']),
        ('', []),
        ('"a long string"', ['"a long string"']),
        ('two\ntokens', ['two', 'tokens']),
        ('one\n\n', ['one']),
        ('"this""is""fine"', ['"this"', '"is"', '"fine"'])
    ]
    for inp, out in examples:
        lexer = Lexer(inp)
        toks = [t[1] for t, lc in lexer.tokenize()]
        assert toks == out

def test_weird_tokens():
    # The lexing rules means that almost anything goes.
    examples = [
        ('\\ \\ \\', ['\\', '\\', '\\']),
        ('\\"', ['\\"'])
        ]
    for inp, out in examples:
        lexer = Lexer(inp)
        toks = [t[1] for t, lc in lexer.tokenize()]
        assert toks == out

def test_line_cols():
    examples = [
        ('hej', [(0, 0)]),
        ('  je', [(0, 2)]),
        ('hej hej', [(0, 0), (0, 4)]),
        ('"" ""', [(0, 0), (0, 3)])
    ]
    for inp, out in examples:
        lexer = Lexer(inp)
        lcs = [lc for t, lc in lexer.tokenize()]
        assert lcs == out

def test_comments():
    examples = [
        (' !not a comment', ['!not', 'a', 'comment']),
        ('!', []),
        ('hey ! these are comments', ['hey'])
        ]
    for inp, out in examples:
        lexer = Lexer(inp)
        toks = [t[1] for t, lc in lexer.tokenize()]
        assert toks == out

def test_errors():
    examples = [
        ('"not finished', (0, 0)),
        # Can't end with an escaped quotation mark
        ('" \\"', (0, 0)),
        # Can't end with a backslash
        ('"  \\', (0, 0))
        ]
    for inp, error_at in examples:
        lexer = Lexer(inp)
        try:
            list(lexer.tokenize())
            assert False
        except LexerError as e:
            assert e.at == error_at

def test_parser():
    examples = [
        (': main ( argc argv -- ) drop ;',
         [
             ('def',
              ('main',
               ((('sym', 'argc'), ('sym', 'argv')), ()),
               (
                   ('sym', 'drop'),
               ),))
         ]),
        (': main ( -- ) ;',
         [
             ('def',
              (
                  'main',
                  ((), ()),
                  (
                  )))
         ]),
        (': main ( -- ) "shit" ;',
         [
             ('def',
              (
                  'main',
                  ((), ()),
                  (
                      ('str', '"shit"'),
                  )))
         ]),
        (': times ( a b -- c ) [ [ [ oooh ] ] ] ;',
         [
             ('def',
              ('times',
               ((('sym', 'a'), ('sym', 'b')), (('sym', 'c'),)),
               (
                   ('quot',
                    (
                        ('quot',
                         (
                             ('quot',
                              (('sym', 'oooh'),)
                             ),)
                        ),)
                   ),)
              )),])
    ]
    for text, expected_tree in examples:
        parser = Parser(Lexer(text))
        tree = parser.parse_defs()
        assert tree == expected_tree

def test_parse_cdef():
    text = 'C: printf4 void printf ( const char* , ... ) 4'
    parser = Parser(Lexer(text))
    tree = parser.parse_defs()
    assert tree == [
        ('cdef',
         ('printf4', 'void', 'printf',
          ('c-args', ['const char*', '...']), 4))
    ]

def test_parse_body():
    text = '10 20 ;'
    parser = Parser(Lexer(text))
    body = parser.parse_body(';')
    assert body == (('int', 10), ('int', 20))
