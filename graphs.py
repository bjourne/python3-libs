from collections import deque
from itertools import islice

def bfs(graph, start, accepting, rep_count):
    queue = deque([[(None, start)]])
    seen = set()
    while queue:
        path = queue.popleft()
        node = path[-1][1]
        if node in accepting:
            tpath = tuple(path)
            if len(tpath) > 1 and tpath not in seen:
                seen.add(tpath)
                yield tpath

        adjacents = graph.get(node, [])
        for trans in adjacents:
            if path.count(trans) < rep_count:
                new_path = list(path)
                new_path.append(trans)
                queue.append(new_path)

def add_transitions(trans, chars, to):
    if len(chars) == 1:
        trans.append((chars, to))
    else:
        for x in range(ord(chars[0]), ord(chars[-1])):
            trans.append((chr(x), to))

def read_ints_line():
    return [int(x) for x in input().split()]

def read_dfa():
    graph = {}
    _, start = read_ints_line()
    accepting = tuple(read_ints_line()[1:])
    trans_count = read_ints_line()[0]
    for x in range(trans_count):
        from_, to, chars = input().split()
        from_ = int(from_)
        to = int(to)

        if from_ not in graph:
            graph[from_] = []
        trans = graph[from_]
        add_transitions(trans, chars, to)
    path_count = read_ints_line()[0]
    return graph, start, accepting, path_count

def main():
    graph, start, accepting, path_count = read_dfa()
    gen = bfs(graph, start, accepting, rep_count = 50)
    paths = list(islice(gen, path_count))
    print(len(paths))
    for path in paths:
        s = ''.join(ch for (ch, _) in path[1:])
        print(s)

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')
