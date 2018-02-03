# 28/36
# 29/36 med len(path)

from collections import deque
from itertools import product
from os import write

def travel(G, start, dests, count):
    queue = deque([[(start, None)]])
    paths = []
    while queue:
        path = queue.popleft()
        at = path[-1][0]
        if at in dests:
            pp = [p[1] for p in path[1:]]
            for real_path in product(*pp):
                paths.append(''.join(real_path))
                count -= 1
                if count == 0:
                    return paths
        adj = G[at]
        for item in adj.items():
            queue.append(path + [item])
    return paths

def add_transitions(trans, chars, to):
    if to not in trans:
        trans[to] = ''
    char_range = range(ord(chars[0]), ord(chars[-1]) + 1)
    extra_chars = ''.join(chr(o) for o in char_range)
    trans[to] = trans[to] + extra_chars

def read_ints_line():
    return [int(x) for x in input().split()]

def read_dfa():
    state_count, start = read_ints_line()
    accepting = set(read_ints_line()[1:])
    trans_count = read_ints_line()[0]
    G = [{} for x in range(state_count)]
    for x in range(trans_count):
        from_, to, chars = input().split()
        from_ = int(from_)
        to = int(to)
        trans = G[from_]
        add_transitions(trans, chars, to)
    path_count = read_ints_line()[0]
    return G, start, accepting, path_count

def main():
    G, start, accepting, path_count = read_dfa()
    paths = travel(G, start, accepting, path_count)
    lines = [str(len(paths))]
    lines.extend(paths)
    lines.append('')
    s = '\n'.join(lines)
    write(1, s.encode('ascii'))

if __name__ == '__main__':
    main()
