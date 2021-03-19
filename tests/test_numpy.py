# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from tools.numpy import slide_window
import numpy as np

def test_slide_window_with_padding():
    A_10 = np.arange(10)
    windows = slide_window(A_10, 3, 3, -1)
    assert np.array_equal(windows, [[0, 1, 2],
                                    [3, 4, 5],
                                    [6, 7, 8],
                                    [9, -1, -1]])

    A_5 = np.arange(5)
    windows = slide_window(A_5, 3, 3, -1)
    assert np.array_equal(windows, [[0, 1, 2],
                                    [3, 4, -1]])

    windows = slide_window(A_5, 3, 2, -1)
    assert np.array_equal(windows, [[0, 1, 2],
                                    [2, 3, 4],
                                    [4, -1, -1]])

    windows = slide_window(A_5, 2, 1, -1)
    assert np.array_equal(windows, [[0, 1],
                                    [1, 2],
                                    [2, 3],
                                    [3, 4],
                                    [4, -1]])

    windows = slide_window(A_5, 5, 5, -1)
    assert np.array_equal(windows, [[0, 1, 2, 3, 4]])

    windows = slide_window(A_10, 2, 2, -1)
    assert np.array_equal(windows, [[0, 1],
                                    [2, 3],
                                    [4, 5],
                                    [6, 7],
                                    [8, 9]])

def test_slide_window_without_padding():
    A_5 = np.arange(5)
    windows = slide_window(A_5, 3, 3, None)
    assert np.array_equal(windows, [[0, 1, 2]])

    A_10 = np.arange(10)
    windows = slide_window(A_10, 3, 3, None)
    assert np.array_equal(windows, [[0, 1, 2],
                                    [3, 4, 5],
                                    [6, 7, 8]])

    windows = slide_window(A_10, 2, 2, None)
    assert np.array_equal(windows, [[0, 1],
                                    [2, 3],
                                    [4, 5],
                                    [6, 7],
                                    [8, 9]])

def test_small_array():
    windows = slide_window(np.arange(5), 10, 10, None)
    assert windows.shape == (0, 10)
    windows = slide_window(np.arange(5), 10, 10, -1)
    assert windows.shape == (1, 10)

def test_big_stuff():
    vs = 100
    A = np.random.randint(vs, size = 100_000)
    delim = np.random.randint(vs)
    I = np.nonzero(A == delim)[0]
    P = np.split(A, I)

    print(P[0].flags['OWNDATA'])

    P = [P[0]] + [p[1:] for p in P[1:]]
    P = [np.array(p) for p in P if len(p) > 0]
    assert len(I) + sum(len(p) for p in P) == len(A)

    for p in P:
        assert not delim in p

    P = [slide_window(p, 50, 49, None) for p in P]
    for p in P:
        for w in p:
            #print(delim, w)
            assert not delim in w
