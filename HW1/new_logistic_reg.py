from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

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
    ell = (1. / n) * np.sum(ells)
    return ell


def calc_grad(X, Y, w):
    n, d = X.shape
    grad = np.zeros(w.shape)
    probs = 1. / (1. + np.exp(-X.dot(w)))
    grad = (1. / n) * (X.T.dot(probs - Y))
    return grad


def new_logistic_regression(X, Y):
    n, d = X.shape
    C = np.concatenate((X, Y.reshape(n, 1)), axis=1) #concatenate X and Y
    C0 = C[C[:, 3] == 0] # points in class 0
    C1 = C[C[:, 3] == 1] # points in class 1
    ctr_C0 = np.mean(C0, axis=0) # center point for class 0
    ctr_C1 = np.mean(C1, axis=0) # center point for class 1
    X_new = np.concatenate((X, ctr_C0[:3].reshape(1, 3), ctr_C1[:3].reshape(1, 3)), axis=0) # add center point to X
    Y_new = np.append(Y, 1, ) # give Y=1 for ctr_C0
    Y_new = np.append(Y_new, 0) # give Y=0 for ctr_C1
    w = np.array([0, 0, 0])
    learning_rate = 10.
    max_iter = 100000

    iter = 0
    while iter < max_iter:
        iter += 1

        # Your code here:
        grad = calc_grad(X_new, Y_new, w)
        w_pre = w
        w = w_pre - learning_rate * grad
        if calc_obj(X_new, Y_new, w) > calc_obj(X_new, Y_new, w_pre):
            learning_rate = learning_rate / 2
            w = w_pre
            print('learning rate shrink to', learning_rate)

        if iter % 1000 == 0:
            print('Finished %d iterations' % iter)
        if np.linalg.norm(calc_grad(X, Y, w)) < 1e-8:
            print('Converged in %d iterations' % iter)
            break
    print(w)
    return


def main():

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('logRegDataB.dat')
    new_logistic_regression(Xb, Yb)

    return


if __name__ == '__main__':
    main()
