# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
from programs.lambdac_golf import *

def test_eval():
    examples = [
        (r'(\f. (\y. f y)) (\x. y) q', 'y'),
        (r'(\n. \s. \z. s (n s z)) (\s. \z. z)',
         r'(\s. (\z. (s z)))'),
        (r'(\n. \s. \z. s (n s z)) (\s. \z. s z)',
         r'(\s. (\z. (s (s z))))'),
        # and true q => q
        (r'(\b. \c. b c (\t. \f. f)) (\t. \f. t) q', 'q'),
        # succ (succ 0) => 2
        (r'(\n. \s. \z. s (n s z)) ((\n. \s. \z. s (n s z)) \s. \z. z)',
         r'(\s. (\z. (s (s z))))'),
        # plus 0 1 => 1
        (r'(\m. \n. \s. \z. m s (n s z)) (\s. \z. z) (\s. \z. s z)',
         r'(\s. (\z. (s z)))'),
        # plus 1 2 => 3
        (r'(\m. \n. \s. \z. m s (n s z)) (\s. \z. s z) (\s. \z. s (s z))',
         r'(\s. (\z. (s (s (s z)))))'),
        (r'(((\x. (\y. y)) (\a. a)) (\b. b))',
         r'(\b. b)'),
        (r'(\a. a) (\x. x) (\y. y)', r'(\y. y)'),

        # alpha-renaming
        (r'(\f. (\x. (f x))) (\y. (\x. y))',  r'(\x. (\y. x))'),

        # From the web page
        (r'((\ x. x) (\ y. (\ z. z)))', r'(\y. (\z. z))')
        ]
    for inp, out in examples:
        #expr = P(inp)
        format_out = E(P(inp))
        if out != format_out:
            print('%s => %s, expected %s' % (inp, format_out, out))
        assert format_out == out
