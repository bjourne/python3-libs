# Copyright (C) 2018 Björn Lindqvist <bjourne@gmail.com>
#
# A naive Bayes classifier, adaptive boosting included. Using
# consistent variable names:
#
#   X: list of samples
#   Y: list of classes for a list of samples
#   W: list of weights for each sample
#   K: list of unique classes
from math import exp
from numpy import (argmax, array,
                   cov,
                   diag, dot,
                   full,
                   histogram,
                   identity,
                   log,
                   shape,
                   unique,
                   zeros)
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
    for x in X:
        vals = [discr(x, p, m, s) for (p, m, s) in zip(prior, mu, sigma)]
        yield argmax(vals, axis = 0)

def init_weights(X):
    N = len(X)
    return full((N,), 1/N)

class BayesClassifier:
    def __init__(self):
        self.prior = None
        self.mu = None
        self.sigma = None

    def fit(self, X, Y, W = None):
        N = len(X)
        if W is None:
            W = init_weights(X)
        self.prior = compute_prior(Y, W)
        self.mu, self.sigma = ml_params(X, Y, W)

    def predict(self, X):
        return classify_bayes(X, self.prior, self.mu, self.sigma)

def train_boost(clf, X, Y, N):
    # Initialize all weights uniformly
    W = init_weights(X)
    insts = []
    alphas = []
    for _ in range(N):
        # Train weak learner
        inst = clf()
        inst.fit(X, Y, W)
        # Get weak hypothesis
        ht = inst.predict(X)

        # Compute its error, et.
        deltas = [0 if v == l else 1 for v, l in zip(ht, Y)]
        et = dot(W, deltas)
        alpha = 0.5 * (log(1 - et) - log(et))

        # Update W
        growth = [exp(-alpha) if d == 0 else exp(alpha) for d in deltas]
        W = W*growth
        # Normalize
        W = W / float(W.sum())
        alphas.append(alpha)
        insts.append(inst)
    return insts, alphas

class BoostClassifier:
    def __init__(self, base_clf, count):
        self.base_clf = base_clf
        self.count = count

    def fit(self, X, Y):
        self.n_classes = len(unique(Y))
        self.insts, self.alphas = train_boost(self.base_clf, X, Y,
                                              self.count)

    def predict(self, X):
        # One hypothesis for each classifier.
        hypothesis = [inst.predict(X) for inst in self.insts]
        for votes in zip(*hypothesis):
            tally = [0] * self.n_classes
            for alpha, vote in zip(self.alphas, votes):
                tally[vote] += alpha
            yield argmax(tally, axis = 0)
