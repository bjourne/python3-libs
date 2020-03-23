# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>

r"""Generation of strings matching regexps.

This module contains a function `inverse_regexp` which takes a regexp
that matches finite length strings and returns all possible matching
string.

The following generates all valid dates in yyyy-mm-dd format for the
year 2020:

    >>> inverse_regexp('2020-'
    ... '(((01|03|05|07|08|10|12)-(0[1-9]|[12][0-9]|3[01]))'
    ... '|02-(0[1-9]|1[0-9]|2[0-9])|(04|06|09|11)-(0[1-9]|[12][0-9]|30))')
    ['2020-01-01', '2020-01-02', '2020-01-03', ...
    '2020-11-28', '2020-11-29', '2020-11-30']

Limitations
===========
Regexps that match infinite sets of strings are not supported:

    >>> inverse_regexp('a*')
    Traceback (most recent call last):
    ...
    ValueError: Star (*) operator is not supported.

This is because the module builds all strings in memory before
returning them. A workaround is to use brace syntax:

    >>> inverse_regexp('a{0,100}')
    ['', 'a', 'aa', 'aaa', 'aaaa', 'aaaaa', ...]

It would be worthwhile to fix the module so that all results are
returned lazily. Then it could support a much larger set of regexps.

Another limitation is that only ASCII is supported:

 * `.` matches any character in `string.printable`.
 * `\d` matches [0-9].
 * `\w` matches [a-zA-Z0-9_]
  * `[^AB] matches any character in `string.printable` except for A and B.

And so on.

Look-ahead and look-behind is for obvious reasons not supported."""
from itertools import chain, product
from sre_constants import *
from sre_parse import parse
from string import ascii_letters, digits, printable, whitespace

PRINTABLE = frozenset(printable)
DIGITS = frozenset(digits)
WHITESPACE = frozenset(whitespace)
LETTERS = frozenset(ascii_letters)
WORDS = DIGITS | LETTERS | {'_'}

CATEGORIES = {
    CATEGORY_DIGIT : DIGITS,
    CATEGORY_NOT_DIGIT : PRINTABLE - DIGITS,
    CATEGORY_NOT_SPACE : PRINTABLE - WHITESPACE,
    CATEGORY_NOT_WORD : PRINTABLE - WORDS,
    CATEGORY_SPACE : WHITESPACE,
    CATEGORY_WORD : WORDS,
}



def handle_any(toks):
    return ALL_CHARACTERS

def handle_branch(toks):
    return frozenset.union(*[handle_toks(tok) for tok in toks[1]])

def handle_category(tok):
    return CATEGORIES[tok]

def handle_in(toks):
    do_negate = toks[0][0] == NEGATE
    if do_negate:
        toks = toks[1:]
    result = frozenset.union(*[handle_tok(tok) for tok in toks])
    if do_negate:
        return ALL_CHARACTERS - result
    return result

def handle_literal(tok):
    return frozenset(chr(tok))

def handle_max_repeat(tok):
    min, max, toks = tok
    assert len(toks) == 1
    if max == MAXREPEAT:
        if min == 0:
            raise ValueError(f'Star (*) operator is not supported.')
        elif min == 1:
            raise ValueError(f'Plus (+) operator is not supported.')

    opt = handle_tok(toks[0])
    opts = frozenset.union(*[frozenset(product(*[opt] * x))
                             for x in range(min, max + 1)])
    return frozenset(''.join(it) for it in opts)

def handle_not_literal(tok):
    return set(printable) - set(chr(tok))

def handle_range(tok):
    lo, hi = tok
    return frozenset(chr(x) for x in range(lo, hi + 1))

def handle_subpattern(tok):
    return handle_toks(tok[3])

HANDLERS = {
    ANY : handle_any,
    BRANCH : handle_branch,
    CATEGORY : handle_category,
    IN : handle_in,
    LITERAL : handle_literal,
    MAX_REPEAT : handle_max_repeat,
    NOT_LITERAL : handle_not_literal,
    RANGE : handle_range,
    SUBPATTERN : handle_subpattern
}

def handle_tok(tok):
    op, arg = tok
    if op not in HANDLERS:
        raise ValueError(f'Unsupported regular expression construct: {op}')
    return HANDLERS[op](arg)

def handle_toks(toks):
    """
    Returns a generator of strings of possible permutations for this
    regexp token list.
    """
    lists = [handle_tok(tok) for tok in toks]
    return frozenset(''.join(it) for it in product(*lists))

def inverse_regexp(s):
    """
    Inverts a regexp.

    >>> for s in inverse_regexp(r'\\d\\w'): print(s)
    0a
    0b
    ...
    9Z
    9_
    """
    toks = parse(s)
    return handle_toks(toks)

if __name__ == '__main__':
    print(inverse_regexp(r'a?'))
