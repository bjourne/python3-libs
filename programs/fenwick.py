# -*- coding: utf-8 -*-
# Copyright (C) 2019 Björn Lindqvist <bjourne@gmail.com>
from sys import stdin, stdout

def main():
    parts = stdin.readline().split()
    n = int(parts[0])
    q = int(parts[1])
    # One extra element so that we can use 1-based indexing.
    nums = [0] * (n + 1)

    outs = []
    app = outs.append
    spl = str.split
    for line in stdin.readlines():
        parts = spl(line)
        if len(parts) == 3:
            i = int(parts[1]) + 1
            inc = int(parts[2])
            while i <= n:
                nums[i] += inc
                i += i & (-i)
        else:
            i = int(parts[1])
            tot = 0
            while i > 0:
                tot += nums[i]
                i -= i & (-i)
            app(tot)
    stdout.write('\n'.join(map(str, outs)) + '\n')

if __name__ == '__main__':
    main()
