# Copyright (C) 2018 Björn Lindqvist
#
# A simple Support Vector Machine written by me, with the help of a
# lot of tutorial!
from numpy import dot, multiply, zeros
from scipy.optimize import minimize

class SVM:
    def __init__(self, kernel, C = 1):
        self.kernel = kernel
        self.C = C

    def kfun(self, x, y):
        if self.kernel == 'linear':
            return dot(x, y)
        elif self.kernel == 'poly':
            return (dot(x, y) + 1)**self.degree
        elif self.kernel == 'radial':
            return -exp(-norm(x - y)**2/(2*self.gamma**2))

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
        n = len(X)
        constraints = {'type' : 'eq', 'fun' : zerofun}
        bounds = [(0, self.C) for b in X]
        start = zeros(n)
        ret = minimize(objective, start,
                       bounds = bounds,
                       constraints = constraints)
        A = ret['x']

        # We select one ARBITRARY support vector
        sv_idxs = [i for (i, a) in enumerate(A) if abs(a) > 0.000001]
        sv_idx = sv_idxs[0]

        sv_x = X[sv_idx]
        sv_y = Y[sv_idx]

        AY = multiply(A, Y)
        AYX = list(zip(AY, X))

        self.b = sum(ay * self.kfun(sv_x, x) for (ay, x) in AYX) - sv_y
        self.W = sum(ay * x for (ay, x) in AYX)

    def predict_point(self, x):
        return self.kfun(self.W, x) - self.b

    def predict(self, points):
        return array([self.predict_point(p) for p in points])
