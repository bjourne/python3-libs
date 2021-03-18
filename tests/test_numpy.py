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
