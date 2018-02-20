from numpy import array
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
    assert abs(svm.b) < 0.001
    assert svm.predict_point([-1,-1]) == -1
    assert svm.predict_point([1, 1]) == 1

if __name__ == '__main__':
    print('hi')
