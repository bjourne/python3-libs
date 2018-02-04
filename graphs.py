from collections import deque
from itertools import product
from os import write

def travel(G, start, accepting, count):
    if start not in G:
        return []
    queue = deque([[(start, None)]])
    paths = []
    while queue:
        path = queue.popleft()
        at = path[-1][0]
        if at in accepting:
            pp = [p[1] for p in path[1:]]
            for real_path in product(*pp):
                paths.append(''.join(real_path))
                count -= 1
                if count == 0:
                    return paths
        for item in G[at]:
            queue.append(path + [item])
    return paths

def add_transitions(trans, chars, to):
    if to not in trans:
        trans[to] = ''
    char_range = range(ord(chars[0]), ord(chars[-1]) + 1)
    trans[to] += ''.join(chr(o) for o in char_range)

def read_ints_line():
    return [int(x) for x in input().split()]

def read_dfa():
    _, start = read_ints_line()
    accepting = set(read_ints_line()[1:])
    G = {}
    trans_count = read_ints_line()[0]
    for x in range(trans_count):
        from_, to, chars = input().split()
        from_ = int(from_)
        to = int(to)
        if from_ not in G:
            G[from_] = {}
        if to not in G:
            G[to] = {}
        trans = G[from_]
        add_transitions(trans, chars, to)
    path_count = read_ints_line()[0]
    return G, start, accepting, path_count

def full_dfa(count):
    from random import choice
    from string import ascii_letters
    G = {}
    for at in range(count):
        G[at] = {}
        for forw_link in range(at + 1, count):
            G[at][forw_link] = choice(ascii_letters)
    return G, 0, set([count - 1]), 500

def prune_redundant_states(G, accepting):
    # First step assigns distances to every state in the dfa.
    distances = {a:0 for a in accepting}
    while True:
        before_len = len(distances)
        for at, trans in G.items():
            if at in distances:
                continue
            for to, chars in trans.items():
                if to in distances:
                    distances[at] = distances[to] + 1
                    break
        if len(distances) == before_len:
            break
    # Second step removes all dead end states from the dfa.
    to_kill = G.keys() - distances.keys()
    for at in to_kill:
        del G[at]
    for at, trans in G.items():
        for t in to_kill:
            trans.pop(t, None)

def main():
    G, start, accepting, path_count = read_dfa()
    prune_redundant_states(G, accepting)
    for at in G:
        G[at] = G[at].items()
    paths = travel(G, start, accepting, path_count)
    lines = [str(len(paths))]
    lines.extend(paths)
    lines.append('')
    s = '\n'.join(lines)
    write(1, s.encode('ascii'))

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()')
    main()
