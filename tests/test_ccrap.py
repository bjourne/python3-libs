from ccrap.lexer import Lexer, LexerError
from ccrap.parser import Parser
from ccrap.typechecker import StackState, combine, format, infer

def test_combine():
    # Corresponds to dup
    state1 = StackState([('dyn', 'a')], [('dyn', 'a'), ('dyn', 'a')])
    state2 = StackState([], [('int', 50)])
    state3 = combine(state1, state2)
    assert state3.input == [('dyn', 'a')]
    assert state3.stack == [('dyn', 'a'), ('dyn', 'b')]

def test_typechecking():
    examples = [
        ('[ swap ]', '( a b -- b a )'),
        ('[ swap swap ]', '( a b -- a b )'),
        ('[ + ]', '( a b -- c )'),
        ('[ 1 swap ]', '( a -- 1 a )'),
        ('[ + - ]', '( a b c -- d )'),
        ('[ [ ] ]', '( -- a )'),

        ('[ nip ]', '( a b -- b )'),
        ('[ dup [ ] dup ]', '( a -- a a b b )'),

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

        # Nested either types
        ('[ 0 0 [ ] [ ] ? [ ] ? call ]', '( -- )'),

        # Dip with either types
        ('[ [ ] [ ] ? dip ]', '( a b -- a )')
    ]
    for inp, expected_out in examples:
        parser = Parser(Lexer(inp))
        quot = parser.parse_token(*parser.next_token())
        out = format(infer(quot[1]))
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
               (('argc', 'argv'), ()),
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
               (('a', 'b'), ('c',)),
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
