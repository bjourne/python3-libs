# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
from math import ceil, floor
import numpy as np

def slide_window(A, win_size, stride, padding = None):
    '''Collects windows that slides over a one-dimensional array.

    If padding is None, the last (rightmost) window is dropped if it
    is incomplete, otherwise it is padded with the padding value.
    '''
    if win_size <= 0:
        raise ValueError('Window size must be positive.')
    if not (0 < stride <= win_size):
        raise ValueError(f'Stride must satisfy 0 < stride <= {win_size}.')

    n_elems = len(A)
    if padding is not None:
        n_windows = ceil(n_elems / stride)
        A = np.pad(A, (0, n_windows * win_size - n_elems),
                   constant_values = padding)
    else:
        n_windows = floor(n_elems / stride)
    shape = n_windows, win_size

    elem_size = A.strides[-1]
    return np.lib.stride_tricks.as_strided(
        A, shape = shape,
        strides = (elem_size * stride, elem_size),
        writeable = False)
