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

def to_string(ast, brackets = False, inleft = False):
    appl_fmt = '%s %s'
    abst_fmt = r'\%s. %s'
    if brackets:
        appl_fmt = '(%s)' % appl_fmt
        abst_fmt = '(%s)' % abst_fmt
    if isinstance(ast, Appl):
        lhs = to_string2(ast.lhs, brackets, True)
        rhs = to_string2(ast.rhs, brackets, inleft)
        if isinstance(ast.rhs, Appl) and not brackets:
            return '%s (%s)' % (lhs, rhs)
        return appl_fmt % (lhs, rhs)
    if isinstance(ast, Abst):
        body = to_string2(ast.body, brackets, inleft)
        if inleft and not brackets:
            return r'(\%s. %s)' % (ast.id, body)
        return abst_fmt % (ast.id, body)
    return ast.id

# See https://www.easycalculation.com/analytical/lambda-calculus.php
def test_parser():
    assert to_string2(parse(r'a (\b. a) c')) == r'a (\b. a) c'
    assert to_string2(parse(r'(\b. a) c')) == r'(\b. a) c'
    print(to_string2(parse('(a b) (c d)')))
    assert to_string2(parse('(a b) (c d)')) == 'a b (c d)'
    assert to_string2(parse('a (b c)')) == 'a (b c)'
    str3 = r'\x. \y. z \m. o'
    assert to_string2(parse(str3)) == r'\x. \y. z \m. o'
    assert to_string2(parse(str3), brackets = True) == r'(\x. (\y. (z (\m. o))))'

    str2 = 'x y z'
    assert to_string2(parse(str2), brackets = True) == '((x y) z)'
    assert to_string2(parse(str2)) == 'x y z'

if __name__ == '__main__':
    test_parser()