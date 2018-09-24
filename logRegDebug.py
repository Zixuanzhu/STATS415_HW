from __future__ import division
import numpy as np

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    n, d = X_.shape
    X = np.zeros((n, d + 1))
    X[:, 0] = 1.
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_obj(X, Y, w):
    n, d = X.shape
    ells = np.log(1. + np.exp(X.dot(w))) - Y * X.dot(w)
    ell  = (1./ n) * np.sum(ells)
    return ell

def calc_grad(X, Y, w):
    n, d = X.shape
    grad = np.zeros(w.shape)
    probs = 1./ (1. + np.exp(-X.dot(w)))
    grad = (1./n) * (X.T.dot(probs - Y))
    return grad

def logistic_regression(X, Y):
    n, d = X.shape
    w = np.zeros(d)
    learning_rate = 10.
    max_iter = 100000

    iter = 0
    while iter < max_iter:
        iter += 1

        """
        YOUR CODE HERE
        """
        
        if iter % 1000 == 0:
            print('Finished %d iterations' % iter)
        if np.linalg.norm(calc_grad(X, Y, w)) < 1e-8:
            print('Converged in %d iterations' % iter)
            break
    return

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('logRegDataA.dat')
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('logRegDataB.dat')
    logistic_regression(Xb, Yb)

    return

if __name__ == '__main__':
    main()
