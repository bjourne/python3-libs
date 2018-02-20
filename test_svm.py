from numpy import array
from numpy.testing import assert_approx_equal
from svm import SVM

def test_init_svm():
    svm = SVM(kernel = 'linear')
    assert svm.kernel == 'linear'
    assert svm.C == 1

def test_kernel():
    svm = SVM(kernel = 'linear')
    assert svm.kfun([10], [10]) == 100

def test_svm_fit():
    svm = SVM(kernel = 'linear')
    X = array([[1,1], [-1,-1]])
    Y = array([1, -1])
    svm.fit(X, Y)
    assert abs(svm.b) < 0.0001
    assert_approx_equal(svm.predict_point([-1,-1]), -1)
    assert_approx_equal(svm.predict_point([1,1]), 1)

from matplotlib import rcParams
from matplotlib.pyplot import (axis,
                               contour,
                               get_current_fig_manager, scatter, show)
from matplotlib.colors import ListedColormap
from numpy import (arange, array,
                   concatenate,
                   linspace,
                   ones, unique)
from numpy.random import randn

def generate_samples(n):
    c1 = [-1, -3]
    c2 = [1, -3]
    c3 = [1.0, 1.0]
    size = 0.3

    p_a1 = randn(int(n / 4), 2) * size + c1
    p_a2 = randn(int(n / 4), 2) * size + c2
    p_a = concatenate([p_a1, p_a2])
    p_b = randn(int(n / 2), 2) * size + c3

    X = concatenate([p_a, p_b])
    Y = concatenate([ones(p_a.shape[0]), -ones(p_b.shape[0])])
    return X, Y

def plot_result(svm, X, Y):
    xg = linspace(-5, 5)
    yg = linspace(-4, 4)

    grid = array([[svm.predict_point([x, y]) for x in xg] for y in yg])
    contour(xg, yg, grid, (-1.0, 0.0, 1.0),
            colors = ('red', 'black', 'blue'),
            linewidths = (1, 3, 1))

    scatter(x = X[Y == -1, 0], y = X[Y == -1, 1], c = 'red')
    scatter(x = X[Y == 1, 0], y = X[Y == 1, 1], c = 'blue')
    axis('equal')
    show()

if __name__ == '__main__':
    # Setting up matplotlib
    rcParams.update({'figure.dpi': 250})
    mng = get_current_fig_manager()
    mng.window.showMaximized()

    X, Y = generate_samples(40)
    svm = SVM('linear', C = 10)
    svm.fit(X, Y)
    plot_result(svm, X, Y)
