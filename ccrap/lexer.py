WS = ' \n'

class LexerError(Exception):
    def __init__(self, message, at):
        super().__init__(message)
        self.at = at

class Lexer:
    def __init__(self, text):
        self.text = text
        self.i = 0
        self.n = len(self.text)
        self.line = 0
        self.col = 0

    # Error handling
    def unfinished_string(self, f, at):
        s = self.text[f:self.i]
        fmt = 'The string `%s` at line %d, column %d didn\'t end ' \
              'with a quotation mark (").'
        message = fmt % (s, at[0] + 1, at[1] + 1)
        raise LexerError(message, at)

    def consume_char(self):
        if self.i == self.n:
            return None
        if self.text[self.i] == '\n':
            self.line += 1
            self.col = 0
        else:
            self.col += 1
        ch = self.text[self.i]
        self.i += 1
        return ch

    def next(self):
        while self.i < self.n and self.text[self.i] in WS:
            self.consume_char()
        if self.i == self.n:
            return None
        f = self.i
        at = self.line, self.col
        if self.text[f] == '"':
            self.consume_char()
            while self.i < self.n and self.text[self.i] not in '"\n':
                if self.consume_char() == "\\":
                    self.consume_char()
            if self.i == self.n or self.text[self.i] == '\n':
                 self.unfinished_string(f, at)
            self.consume_char()
        else:
            while self.i < self.n and self.text[self.i] not in WS:
                self.consume_char()
        tok = self.text[f:self.i]
        if tok == '!':
            while self.i < self.n and self.text[self.i] != '\n':
                self.consume_char()
            return self.next()
        return tok, at

    def tokenize(self):
        while True:
            tok = self.next()
            if tok is None:
                return
            yield tok

if __name__ == '__main__':
    text = """
: printf4
    swap { void printf ( const char* , ... ) 4 } ;
"""

    lexer = Lexer(text)
    for t in lexer.tokenize():
        print('EMIT', t)
