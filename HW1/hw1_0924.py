import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

##################1.(c)
data = np.loadtxt('locLinRegData.txt')
label = np.array(data[:, 0])
y = np.reshape(label, (np.shape(data[:, 0])[0], 1))
feature = np.array(data[:, 1])
x = np.reshape(feature, (np.shape(data[:, 1])[0], 1))
#In LM, beta = (X^T X)^(-1) X^T y
x0 = np.ones((np.shape(x)))
X = np.append(x0, x, axis= 1)
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

a = np.linspace(-5, 15, 100)
b = beta[0] + beta[1]*a
plt.scatter(feature, label)
plt.plot(a, b)
plt.show()

###############1.(e)
test_data = np.linspace(-5, 15, 100)
test_y = []
tau = 0.8

#In question, we've already have beta=(X^T W X)^{-1} X^T W y
for x in test_data:
    w = np.exp(- (feature - x) ** 2 / (2*tau ** 2))
    W = np.diagflat(w)
    beta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
    test_y.append(beta[0] + beta[1]*x)

plt.scatter(feature, label)
plt.plot(test_data, test_y)
plt.show()

###############1.(f)
taus = [0.1, 0.3, 2, 10]
for i in range(4):
    test_data = np.linspace(-5, 13, 100)
    test_y = []
    for x in test_data:
        w = np.exp(- (feature - x) ** 2 / (2 * taus[i] ** 2))
        W = np.diagflat(w)
        beta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
        test_y.append(beta[0] + beta[1] * x)

    plt.scatter(feature, label)
    plt.plot(test_data, test_y)
    plt.title('bandwidth = ' + str(taus[i]))
    plt.show()
