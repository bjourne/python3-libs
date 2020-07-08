from learning.tensorflow import skew_top_p
import numpy as np

def test_skew_top_p():
    P = np.array([0.1, 0.1, 0.8])
    P = skew_top_p(P, 0.8)
    assert P.sum() == 1.0
    assert P[2] == 1.0

    P = np.array([0.3, 0.3, 0.4])
    P2 = skew_top_p(P, 0.8)
    assert P2.sum() == 1.0
    assert np.array_equal(P, P2)

    P = np.array([0.4, 0.4, 0.2])
    P2 = skew_top_p(P, 0.8)
    assert P2.sum() == 1.0
    assert np.array_equal(P2, [0.5, 0.5, 0.0])

def test_skew_top_p_float_prec():
    P = np.array([0.1] * 10)
    P2 = skew_top_p(P, 0.9)
    assert P2.sum() == 1.0

    # Due to float precision issues.
    assert np.array_equal(P, P2)

def test_skew_top_p_small_probs():
    P = np.array([0.05] * 20)
    P2 = skew_top_p(P, 0.9)
    assert P2.sum() == 1.0
    assert P2[-1] == 0.0
    assert P2[-2] == 0.0
    assert round(P2[0], 4) == round(0.05 / 0.9, 4)
