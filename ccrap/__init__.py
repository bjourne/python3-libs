# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
#
# This is a toy language for writing C code in concatenative
# style. You run it like this:
#
#     python -m ccrap.__init__ -f test.ccrap \
#         | gcc -O3 -o main -xc - && ./main
#
# It is not meant to be taken seriously (yet!).
#
# FAQ
# ===
#
# Why so many intrinsics?
# -----------------------
# It is an easy way to improve the code clang and gcc emits.
#
# What is missing?
# ----------------
# More things
#
# Types?
# ------
# Not yet, but my ambition is to implement type inferencing.
from argparse import ArgumentParser
from codecs import open
from ccrap.codegen import generate_vocab
from ccrap.lexer import Lexer
from ccrap.optimizer import optimize
from ccrap.parser import Parser

if __name__ == '__main__':
    parser = ArgumentParser(description = 'CCrap compiler')
    parser.add_argument('--file', '-f',
                        type = str, required = True,
                        help = 'File to compile')
    args = parser.parse_args()
    with open(args.file, 'r', 'utf-8') as f:
        text = f.read()
    parser = Parser(Lexer(text))
    defs = parser.parse_defs()
    defs = optimize(defs)
    text = generate_vocab(defs)
    print(text)
