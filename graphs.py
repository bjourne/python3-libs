from collections import deque
from itertools import islice, product

def bfs(graph, start, accepting):
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
            new_path = list(path)
            new_path.append(trans)
            queue.append(new_path)

def add_transitions(trans, chars, to):
    gen = range(ord(chars[0]), ord(chars[-1]) + 1)
    trans.append((gen, to))

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

def generate_paths(path_gen):
    for path_pattern in path_gen:
        gens = [g[0] for g in path_pattern[1:]]
        for path in product(*gens):
            yield ''.join(chr(o) for o in path)

def main():
    graph, start, accepting, path_count = read_dfa()
    gen = bfs(graph, start, accepting)
    paths = islice(generate_paths(gen), path_count)
    paths = list(paths)
    print(len(paths))
    for path in paths:
        print(path)

if __name__ == '__main__':
    # main()
    import cProfile
    cProfile.run('main()')
