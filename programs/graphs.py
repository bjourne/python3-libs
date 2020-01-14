from collections import deque, defaultdict
from itertools import islice, product

def travel(G, start, accepting):
    if start not in G:
        return []
    queue = deque([[(start, None)]])
    while queue:
        path = queue.popleft()
        at = path[-1][0]
        if at in accepting:
            pp = [p[1] for p in path[1:]]
            for real_path in product(*pp):
                yield ''.join(real_path)
        for item in G[at].items():
            queue.append(path + [item])

def read_ints_line():
    return [int(x) for x in input().split()]

def read_transition(G):
    from_, to, chars = input().split()
    from_ = int(from_)
    to = int(to)
    char_range = range(ord(chars[0]), ord(chars[-1]) + 1)
    char_str = ''.join(chr(o) for o in char_range)
    G[from_][to] += char_str

def read_dfa():
    _, start = read_ints_line()
    accepting = set(read_ints_line()[1:])
    G = defaultdict(lambda: defaultdict(str))
    trans_count = read_ints_line()[0]
    for x in range(trans_count):
        read_transition(G)
    path_count = read_ints_line()[0]
    return G, start, accepting, path_count

def prune_redundant_states(G, accepting):
    # A set of nodes which has a path to the accepting nodes.
    reachable = set(accepting)
    while True:
        prev_count = len(reachable)
        # Iterate the graph and see if we find any that leads to the
        # reachable nodes.
        for at, trans in G.items():
            if at in reachable:
                continue
            for to in trans:
                if to in reachable:
                    reachable.add(at)
                    break
        # Done because no extra reachable nodes found during the
        # iteration.
        if len(reachable) == prev_count:
            break

    # Second step removes all dead end states from the graph.
    to_kill = G.keys() - reachable
    for at in to_kill:
        del G[at]
    for at, trans in G.items():
        for t in to_kill:
            trans.pop(t, None)

def main():
    G, start, accepting, count = read_dfa()
    prune_redundant_states(G, accepting)
    paths = list(islice(travel(G, start, accepting), count))
    print(len(paths))
    for path in paths:
        print(path)

if __name__ == '__main__':
    main()
