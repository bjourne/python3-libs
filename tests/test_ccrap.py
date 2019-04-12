from ccrap.lexer import Lexer, LexerError

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
        toks = [t for t, lc in lexer.tokenize()]
        assert toks == out

def test_weird_tokens():
    # The lexing rules means that almost anything goes.
    examples = [
        ('\\ \\ \\', ['\\', '\\', '\\']),
        ('\\"', ['\\"'])
        ]
    for inp, out in examples:
        lexer = Lexer(inp)
        toks = [t for t, lc in lexer.tokenize()]
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
        toks = [t for t, lc in lexer.tokenize()]
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
