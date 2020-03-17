# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Simple function for validating Swedish personal numbers.
def validate(pnr):
    '''
    pnr should be a string on the format 'yymmdd-xxxx'
    '''
    first6 = pnr[:6]
    digits = [int(d) for d in pnr[:6] + pnr[-4:]]

    even_digitsum = sum(x if x < 5 else x - 9 for x in digits[::2])
    return sum(digits, even_digitsum) % 10 == 0
