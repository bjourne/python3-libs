# Copyright (C) 2018 Björn Lindqvist <bjourne@gmail.com>
#
# Based on:
# https://tadeuzagallo.com/blog/writing-a-lambda-calculus-interpreter-in-javascript/
from collections import namedtuple
from re import findall

LAMBDA = 'lambda'
DOT = '.'
LPAREN = '('
RPAREN = ')'
LCID = 'lcid'
EOF = 'eof'

Token = namedtuple('Token', ['type', 'value'])

class Lexer:
    def __init__(self, str):
        self.input = findall(r'(\(|\)|λ|\\|[a-z][a-z]*|\.)', str)
        self.index = 0
        self.token = None
        self.next_token()

    def next_token(self):
        if not self.input:
            self.token = Token(EOF, None)
            return
        at = self.input.pop(0)
        if at in r'λ\\':
            self.token = Token(LAMBDA, None)
        elif at == '.':
            self.token = Token(DOT, None)
        elif at == '(':
            self.token = Token(LPAREN, None)
        elif at == ')':
            self.token = Token(RPAREN, None)
        else:
            self.token = Token(LCID, at)

    def next(self, type):
        return self.token.type == type

    def skip(self, type):
        if self.next(type):
            self.next_token()
            return True
        return False

    def match(self, type):
        if self.next(type):
            self.next_token()
            return
        raise Exception('Parse error!')

    def value(self, type):
        '''
        Returns value of current token.
        '''
        tok = self.token
        self.match(type)
        return tok.value

Ident = namedtuple('Ident', ['id'])
Appl = namedtuple('Appl', ['lhs', 'rhs'])
Abst = namedtuple('Abst', ['id', 'body'])

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer

    def parse(self):
        result = self.term()
        self.lexer.match(EOF)
        return result

    def term(self):
        if self.lexer.skip(LAMBDA):
            id = self.lexer.value(LCID)
            self.lexer.match(DOT)
            term = self.term()
            return Abst(id, term)
        return self.appl()

    def appl(self):
        lhs = self.atom()
        while True:
            rhs = self.atom()
            if not rhs:
                return lhs
            lhs = Appl(lhs, rhs)

    def atom(self):
        if self.lexer.skip(LPAREN):
            term = self.term()
            self.lexer.match(RPAREN)
            return term
        elif self.lexer.next(LCID):
            id = self.lexer.value(LCID)
            return Ident(id)
        elif self.lexer.skip(LAMBDA):
            id = self.lexer.value(LCID)
            self.lexer.match(DOT)
            term = self.term()
            return Abst(id, term)
        return None

def parse(str):
    return Parser(Lexer(str)).parse()

def to_string(ast, brackets = False):
    appl_fmt = '%s %s'
    abst_fmt = r'\%s. %s'
    if brackets:
        appl_fmt = '(%s)' % appl_fmt
        abst_fmt = '(%s)' % abst_fmt
    if isinstance(ast, Appl):
        str_lhs = to_string(ast.lhs, brackets)
        str_rhs = to_string(ast.rhs, brackets)
        if isinstance(ast.rhs, Appl) and not brackets:
            str_rhs = '(%s)' % str_rhs
        if isinstance(ast.lhs, Abst) and not brackets:
            str_lhs = '(%s)' % str_lhs
        return appl_fmt % (str_lhs, str_rhs)
    if isinstance(ast, Ident):
        return ast.id
    if isinstance(ast, Abst):
        body = to_string(ast.body, brackets)
        return abst_fmt % (ast.id, body)

# See https://www.easycalculation.com/analytical/lambda-calculus.php
def test_parser():
    str1 = r'\x. x'
    assert to_string(parse(str1)) == r'\x. x'
    assert str(parse(str1)) == "Abst(id='x', body=Ident(id='x'))"
    str2 = 'x y z'
    to_string(parse(str2), brackets = True)
    assert to_string(parse(str2), brackets = True) == '((x y) z)'
    assert to_string(parse(str2)) == 'x y z'
    str3 = r'\x. \y. z \m. o'
    assert to_string(parse(str3)) == r'\x. \y. z \m. o'
    assert to_string(parse(str3), brackets = True) == r'(\x. (\y. (z (\m. o))))'

    str4 = r'a (b c)'
    assert to_string(parse(str4)) == 'a (b c)'

    # Since we are parsing from the left
    assert to_string(parse('(a b) (c d)')) == 'a b (c d)'

    # Brackets to disambiguate
    assert to_string(parse(r'(\b. a) c')) == r'(\b. a) c'

    # Need brackets here too, doesn't work yet.
    assert to_string(parse(r'a (\b. a) c')) == r'a (\b. a) c'


if __name__ == '__main__':
    test_parser()
