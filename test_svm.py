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
                               clf, contour,
                               figure,
                               get_current_fig_manager,
                               savefig, scatter, show,
                               title)
from numpy import (arange, array,
                   concatenate,
                   linspace,
                   ones, unique)
from numpy.random import randn, seed
from random import randint

v = randint(1, 100000)
print(v)
seed(v)

def generate_samples(n, clusters):
    c1, c2, c3 = clusters
    size = 1.2

    p_a1 = randn(int(n / 4), 2) * size + c1
    p_a2 = randn(int(n / 4), 2) * size + c2
    p_a = concatenate([p_a1, p_a2])
    p_b = randn(int(n / 2), 2) * size + c3

    X = concatenate([p_a, p_b])
    Y = concatenate([ones(p_a.shape[0]), -ones(p_b.shape[0])])
    return X, Y

def plot_result(svm, X, Y):
    figure()
    def do_predict(x, y):
        inp = [[x, y]]
        return svm.predict(inp)[0]

    xr = (X[:,0].min() - 1, X[:,0].max() + 1)
    yr = (X[:,1].min() - 1, X[:,1].max() + 1)

    xg = linspace(*xr)
    yg = linspace(*yr)

    grid = array([[do_predict(x, y) for x in xg] for y in yg])
    contour(xg, yg, grid, (-1.0, 0.0, 1.0),
            colors = ('red', 'black', 'blue'),
            linewidths = (1, 3, 1))

    scatter(x = X[Y == -1, 0], y = X[Y == -1, 1], c = 'red')
    scatter(x = X[Y == 1, 0], y = X[Y == 1, 1], c = 'blue')

def make_svm_plot(n, clusters, X, Y,
                  ktype, C, degree = 3, coef0 = 1, gamma = 2):
    filename_pre = ktype
    title_pre = ktype
    if ktype == 'linear':
        name = '%s (C = %.1f' % (title_pre, C)
        fname = '%s-%.1f.png' % (filename_pre, C)
    elif ktype == 'rbf':
        name = '%s (C = %.1f, gamma = %.1f' % (title_pre, C, gamma)
        fname = '%s-%.1f-%.1f.png' % (filename_pre, C, gamma)
    elif ktype == 'poly':
        name = '%s (C = %.1f, degree = %d, coef0 = %d' \
                        % (title_pre, C, degree, coef0)
        fname = '%s-%.1f-%d-%d.png' % (filename_pre, C, degree, coef0)
    name += ' clusters=%s)' % clusters
    # X, Y = generate_samples(n, clusters)
    svm = SVM(ktype,
              C = C, degree = degree, coef0 = coef0, gamma = gamma)
    svm.fit(X, Y)
    plot_result(svm, X, Y)
    title(name)
    savefig(fname)
    clf()


if __name__ == '__main__':
    # Setting up matplotlib
    rcParams.update({'figure.dpi': 250})
    mng = get_current_fig_manager()
    mng.window.showMaximized()

    c1 = [-4, -2]
    c2 = [ 4, -2]
    c3 = [ 4,  4]
    clusters = [c1, c2, c3]
    n = 50

    X, Y = generate_samples(n, clusters)
    make_svm_plot(n, [c1, c2, c3], X, Y, 'linear', 0.1)
    make_svm_plot(n, [c1, c2, c3], X, Y, 'linear', 1.0)
    make_svm_plot(n, [c1, c2, c3], X, Y, 'linear', 5.0)
    make_svm_plot(n, [c1, c2, c3], X, Y, 'linear', 20.0)
