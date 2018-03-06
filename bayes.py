# Copyright (C) 2018 Björn Lindqvist <bjourne@gmail.com>
#
# A naive Bayes classifier, adaptive boosting included. Using
# consistent variable names:
#
#   X: list of samples
#   Y: list of classes for a list of samples
#   W: list of weights for each sample
#   K: list of unique classes
from numpy import (argmax, array,
                   cov, diag, full,
                   identity,
                   log,
                   unique)
from numpy.linalg import det, inv

def make_cov(X, W):
    """Weighted and biased covariance matrix assuming feature
    independence."""
    M = cov(X.T, bias = True, aweights = W)
    D = diag(M)
    return identity(len(D)) * D

def ml_params(X, Y, W):
    classes = unique(Y)
    X_groups = array([X[Y == i] for i in classes])
    W_groups = array([W[Y == i] for i in classes])
    mu = array([(W.reshape(1, -1).T * X).sum(0) / W.sum()
                for X, W in
                zip(X_groups, W_groups)])
    sigma = [make_cov(X, W)
             for X, W in
             zip(X_groups, W_groups)]
    return mu, sigma

def compute_prior(Y, W):
    classes = unique(Y)
    W_groups = array([W[Y == i] for i in classes])
    s = array([wg.sum() for wg in W_groups])
    return s / s.sum()

def discr(x, prior, mu, sigma):
    """One mu and one sigma -- not a list.
    """
    return -(1/2)*log(det(sigma)) \
        -(1/2)*(x - mu).dot(inv(sigma)).dot(x - mu) \
        + log(prior)

def classify_bayes(X, prior, mu, sigma):
    hs = []
    for x in X:
        vals = [discr(x, p, m, s) for (p, m, s) in zip(prior, mu, sigma)]
        h = argmax(vals, axis = 0)
        hs.append(h)
    return hs

class BayesClassifier:
    def __init__(self):
        self.prior = None
        self.mu = None
        self.sigma = None

    def fit(self, X, Y, W = None):
        N = len(X)
        if W is None:
            W = full((N,), 1/N)
        self.prior = compute_prior(Y, W)
        self.mu, self.sigma = ml_params(X, Y, W)

    def predict(self, X):
        return classify_bayes(X, self.prior, self.mu, self.sigma)
