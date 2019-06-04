from ccrap.lexer import Lexer

class ParserError(Exception):
    def __init__(self, message, at):
        super().__init__(message)
        self.at = at

def format_any_expected(alts):
    alts = ['`%s`' % e for e in alts]
    n = len(alts)
    if n == 1:
        return alts[0]
    elif n == 2:
        return '%s or %s' % tuple(alts)
    return '%s or %s' % (', '.join(alts[:-1]), alts[-1])

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.at = (0, 0)

    # Error handling
    def wrong_token(self, got, exp_str):
        fmt = 'Got `%s` at line %d, column %d, expected %s.'
        line = self.at[0] + 1
        col = self.at[1] + 1
        message = fmt % (got, line, col, exp_str)
        raise ParserError(message, self.at)

    def end_of_file(self, exp_str):
        fmt = 'End of file at line %d, column %d, expected %s.'
        line = self.at[0] + 1
        col = self.at[1] + 1
        message = fmt % (line, col, exp_str)
        raise ParserError(message, self.at)

    # Convenient parsing methods
    def next_token(self):
        tok = self.lexer.next()
        if tok:
            self.at = tok[1]
            return tok[0]
        return None

    def match_token(self, exp):
        tok = self.next_token()
        if not tok:
            self.at = self.lexer.line, self.lexer.col
            self.end_of_file('`%s`' % exp)
        type, val = tok
        if val != exp:
            self.wrong_token(val, '`%s`' % exp)
        return val

    def parse_until_any(self, any_of):
        toks = []
        while True:
            tok = self.next_token()
            if not tok:
                expected = format_any_expected(any_of)
                self.end_of_file(expected)
            type, val = tok
            if val in any_of:
                return val, toks
            toks.append(val)

    def parse_until(self, until):
        return self.parse_until_any([until])[1]

    def parse_int(self):
        _, v = self.next_token()
        try:
            return int(v)
        except ValueError:
            self.wrong_token(v, 'an integer')

    def parse_effect(self):
        self.match_token('(')
        lhs = tuple(self.parse_until('--'))
        rhs = tuple(self.parse_until(')'))
        return (lhs, rhs)

    def parse_token(self, type, val):
        if val == '[':
            return 'quot', self.parse_body(']')
        elif val.isdigit():
            return 'int', int(val)
        return type, val

    def parse_body(self, stop_sym):
        """Parses definition bodies and quotations."""
        body = []
        while True:
            tok = self.next_token()
            if not tok:
                self.end_of_file('`%s`' % stop_sym)
            type, val = tok
            if val == stop_sym:
                return tuple(body)
            body.append(self.parse_token(type, val))

    def parse_def(self):
        _, name = self.next_token()
        effect = self.parse_effect()
        body = self.parse_body(';')
        return 'def', (name, effect, body)

    def parse_cdef_args(self):
        self.match_token('(')
        args = []
        while True:
            last, arg = self.parse_until_any([')', ','])
            if arg:
                args.append(' '.join(arg))
            if last == ')':
                return 'c-args', args

    def parse_cdef(self):
        _, name = self.next_token()
        _, ret = self.next_token()
        _, c_name = self.next_token()
        c_args = self.parse_cdef_args()
        n_var_args = self.parse_int()
        if n_var_args < 0:
            self.wrong_token(n_args, 'a non-negative integer')
        return 'cdef', (name, ret, c_name, c_args, n_var_args)

    def parse_defs(self):
        defs = []
        def_handlers = {
            ':' : self.parse_def,
            'C:' : self.parse_cdef
            }

        while True:
            tok = self.next_token()
            if not tok:
                return defs
            _, val = tok
            if val is None:
                return defs
            elif val in def_handlers:
                defs.append(def_handlers[val]())
            else:
                expected = format_any_expected(def_handlers.keys())
                self.wrong_token(val, expected)
        return 'vocab', defs

if __name__ == '__main__':
    text = """
: times ( a b -- c ) oooh ;
"""
    parser = Parser(Lexer(text))
    print(parser.parse_defs())
