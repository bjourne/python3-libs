# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from tools.inverse_regexp import *

def test_simple_in():
    results = inverse_regexp(r'\d\w')
    cat_word = CATEGORIES[CATEGORY_WORD]
    cat_digit = CATEGORIES[CATEGORY_DIGIT]
    assert len(results) == len(cat_word) * len(cat_digit)

def test_spaces():
    results = inverse_regexp(r'\s')
    cat_spaces = CATEGORIES[CATEGORY_SPACE]
    assert len(results) == len(cat_spaces)

def test_range():
    results = inverse_regexp(r'[0-9]')
    assert len(results) == 10

def test_subpattern():
    results = inverse_regexp(r'([0-9])')
    assert len(results) == 10

def test_literal():
    results = inverse_regexp(r'aaa')
    assert len(results) == 1

def test_branch():
    results = inverse_regexp(r'a(a|b(c|d))')
    assert results == {'aa', 'abc', 'abd'}

def test_max_repeat():
    results = inverse_regexp(r'a{1,3}')
    assert results == {'a', 'aa', 'aaa'}

def test_star():
    try:
        inverse_regexp(r'a*')
        assert False
    except ValueError:
        assert True

def test_duplicates():
    results = inverse_regexp(r'(A|A)')
    assert results == {'A'}

def test_not_literal():
    results = inverse_regexp(r'[^A]')
    assert 'A' not in results
