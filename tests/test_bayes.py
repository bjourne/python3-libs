from learning.bayes import BayesClassifier, BoostClassifier
from numpy import allclose, array, array_equal
from sklearn.datasets import load_iris

d = load_iris()
irisX = d['data']
irisY = d['target']

def test_naive_bayes():
    bc = BayesClassifier()
    assert bc.prior is None
    bc.fit(irisX, irisY)
    assert array_equal(bc.prior, [1/3, 1/3, 1/3])
    assert allclose(bc.mu, [[5.006, 3.428, 1.462, 0.246],
                            [5.936, 2.77,  4.26,  1.326],
                            [6.588, 2.974, 5.552, 2.026]])

def test_classifying():
    bc = BayesClassifier()
    bc.fit(irisX, irisY)
    irisPredY = list(bc.predict(irisX))
    # Only six mismatches
    assert sum(irisY == irisPredY) == 144

def test_boosting():
    bc = BoostClassifier(BayesClassifier, 10)
    bc.fit(irisX, irisY)
    expected = [1.589, 1.022, 0.591, 1.398, 0.637,
                0.276, -0.242, 0.081, -0.072, 0.002]
    assert allclose(bc.alphas, expected, atol = 10e-3)
    irisPredY = list(bc.predict(irisX))
    assert sum(irisY == irisPredY) == 147

def test_boosting_one():
    bc = BoostClassifier(BayesClassifier, 1)
    bc.fit(irisX, irisY)
    irisPredY = list(bc.predict(irisX))
    assert sum(irisY == irisPredY) == 144


if __name__ == '__main__':
    bc = BayesClassifier()
    bc.fit(irisX, irisY)
