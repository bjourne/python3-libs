# Copyright (C) 2018-2019 Björn Lindqvist <bjourne@gmail.com>
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
    def __init__(self, str):
        self.tokens = findall(r'(\(|\)|λ|\\|[a-z][a-z]*|\.)', str)
        #self.lexer = lexer

    def parse(self):
        result = self.term()
        self.lexer.match(EOF)
        return result

    def term(self):
        if self.tokens[0].type == LAMBDA:
            self.tokens.pop(0)
            lcid = self.topens.pop(0)
            assert lcid.type == LCID
            assert self.tokens.pop(0).type == DOT
            term = self.term()
            return Abst(lcid.id, term)
        return self.appl()
        # if self.lexer.skip(LAMBDA):
        #     id = self.lexer.value(LCID)
        #     self.lexer.match(DOT)
        #     term = self.term()
        #     return Abst(id, term)
        # return self.appl()

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
    return Parser(str).parse()
    #return Parser(Lexer(str)).parse()

def to_string(ast, brackets = False, inleft = False):
    appl_fmt = '%s %s'
    abst_fmt = r'\%s. %s'
    if brackets:
        appl_fmt = '(%s)' % appl_fmt
        abst_fmt = '(%s)' % abst_fmt
    if isinstance(ast, Appl):
        lhs = to_string(ast.lhs, brackets, True)
        rhs = to_string(ast.rhs, brackets, inleft)
        if isinstance(ast.rhs, Appl) and not brackets:
            return '%s (%s)' % (lhs, rhs)
        return appl_fmt % (lhs, rhs)
    if isinstance(ast, Abst):
        params = [ast.id]
        body = ast.body
        while isinstance(body, Abst):
            params += [body.id]
            body = body.body
        body = to_string(body, brackets, False)
        param_str = ' '.join(params)
        if inleft and not brackets:
            return r'(\%s. %s)' % (param_str, body)
        return abst_fmt % (param_str, body)
    return ast.id

# See https://www.easycalculation.com/analytical/lambda-calculus.php
def test_to_string():
    assert to_string(parse(r'a (\b. a) c')) == r'a (\b. a) c'
    assert to_string(parse(r'(\b. a) c')) == r'(\b. a) c'
    assert to_string(parse('(a b) (c d)')) == 'a b (c d)'
    assert to_string(parse('a (b c)')) == 'a (b c)'
    str3 = r'\x. \y. z \m. o'
    assert to_string(parse(str3)) == r'\x y. z \m. o'
    assert to_string(parse(str3), brackets = True) \
        == r'(\x y. (z (\m. o)))'

    s = 'x y z'
    assert to_string(parse(s), brackets = True) == '((x y) z)'
    assert to_string(parse(s)) == 'x y z'

    examples = [
        (r'(\b. \c. b c (\t. \f. f)) (\t. \f. f)',
         r'(\b c. b c \t f. f) \t f. f'),
        (r'(\n. \s. \z. s (n s z)) (\s. \z. s z)',
         r'(\n s z. s (n s z)) \s z. s z'),
        (r'(\x. x) (\y. y)',
         r'(\x. x) \y. y'),
        (r'((a (\x. x)) a)', r'a (\x. x) a')
        ]
    for inp, out in examples:
        assert to_string(parse(inp)) == out


if __name__ == '__main__':
    Parser(r'\x. x').parse()
    #test_to_string()
