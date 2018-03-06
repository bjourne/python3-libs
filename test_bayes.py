from bayes import BayesClassifier
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
    assert allclose(bc.mu, [[5.006, 3.418, 1.464, 0.244],
                            [5.936, 2.77,  4.26,  1.326],
                            [6.588, 2.974, 5.552, 2.026]])

def test_classifying():
    bc = BayesClassifier()
    bc.fit(irisX, irisY)
    irisPredY = bc.predict(irisX)
    # Only six mismatches
    assert sum(irisY == irisPredY) == 144

if __name__ == '__main__':
    bc = BayesClassifier()
    bc.fit(irisX, irisY)
