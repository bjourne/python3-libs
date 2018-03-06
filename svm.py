# Copyright (C) 2018 Björn Lindqvist <bjourne@gmail.com>
#
# A simple Support Vector Machine written by me, with the help of a
# lot of tutorials!
from math import exp
from numpy import array, dot, multiply, nonzero, zeros
from numpy.linalg import norm
from scipy.optimize import minimize

class SVM:
    def __init__(self, kernel, C = 1,
                 degree = 2, coef0 = 1.0,
                 gamma = 0.1):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def kfun(self, x, y):
        if self.kernel == 'linear':
            return dot(x, y)
        elif self.kernel == 'poly':
            return (dot(x, y) + self.coef0)**self.degree
        elif self.kernel == 'rbf':
            return exp(-dot(x - y, x - y)/(2*self.gamma**2))

    def fit(self, X, Y):
        """Trains the SVM. X is a sequence of vectors of the
        samples features and Y a sequence of scalar classes,
        often -1 and 1.

        For clarity(?), I have used capital names for sequences and
        lowercase for individual elements. X are the features, Y the
        classes and A the alphas.
        """
        def zerofun(A):
            return dot(A, Y)
        def objective(A):
            v = 0
            for a_i, x_i, y_i in zip(A, X, Y):
                for a_j, x_j, y_j in zip(A, X, Y):
                    v += a_i * a_j * y_i * y_j * self.kfun(x_i, x_j)
            return (v / 2) - sum(A)
        constraints = {'type' : 'eq', 'fun' : zerofun}
        bounds = [(0, self.C) for b in X]
        start = zeros(len(X))
        ret = minimize(objective, start,
                       bounds = bounds,
                       constraints = constraints)
        A = ret['x']

        # Select the non-zero support vectors
        indices = [i for (i, a) in enumerate(A) if a > 0.000001]
        A = A[indices]
        Y = Y[indices]
        X = X[indices]
        self.AYX = list(zip(multiply(A, Y), X))
        self.b = sum(ay * self.kfun(X[0], x)
                     for (ay, x) in self.AYX) - Y[0]

    def predict_point(self, x):
        return sum(ay * self.kfun(x, xi)
                   for (ay, xi) in self.AYX) - self.b

    def predict(self, points):
        return array([self.predict_point(p) for p in points])
