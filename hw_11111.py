import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


a = 3
b = 4
c = a + b
c += 1
d = c


data_p = pd.read_table('locLinRegData.txt')
n = np.shape(data)[0]
feature = data[:,1].reshape(n,1)
label = data[:,0].reshape(n,1)
ones = np.ones(shape=(len(feature),1))
X = np.append(ones, feature, axis=1)
y = label
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

a = np.linspace(-5,13,100)
b = beta[0] + beta[1]*a

plt.scatter(feature,label)
plt.plot(a, b)
plt.show()
