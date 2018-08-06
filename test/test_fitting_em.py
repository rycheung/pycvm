import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import fitting
import numpy as np
import matplotlib.pyplot as plt


def gaussian(X, mu, sig):
    k = 1 / ((2 * np.pi) ** (mu.size / 2) * np.sqrt(np.linalg.det(sig)))
    sig_inv = np.linalg.pinv(sig)
    exp_factors = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        exp_factors[i] = np.exp(-0.5 * np.dot(np.dot((X[i] - mu),
                                                     sig_inv), np.transpose([X[i] - mu])))
    return k * exp_factors


I = 10000
K = 2

p_lambda1 = 0.5
mu1 = np.array([1, 2])
sig1 = np.array([[2, 0], [0, 0.5]])

p_lambda2 = 0.5
mu2 = np.array([1, 5])
sig2 = np.array([[1, 0], [0, 1]])

X1 = np.random.multivariate_normal(mu1, sig1, round(I * p_lambda1))
X2 = np.random.multivariate_normal(mu2, sig2, round(I * p_lambda2))
X = np.append(X1, X2, axis=0)
I = X.shape[0]

p_lambda, mu, sig = fitting.em_mog(X, K, 0.01)

XX, YY = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))
x = XX.reshape((XX.size, 1))
y = YY.reshape((XX.size, 1))
xy_matrix = np.append(x, y, axis=1)
mog = np.zeros((XX.size, 1))
for k in range(K):
    mog[:, 0] += p_lambda[k] * gaussian(xy_matrix, mu[k], sig[k])[:, 0]
mog = mog.reshape((XX.shape[0], XX.shape[1]))

plt.figure("EM MoG")
plt.subplot(211)
plt.scatter(X[:, 0], X[:, 1], marker=".")
plt.axis([-6, 6, -1, 9])
plt.contour(XX, YY, mog)

XX, YY = np.meshgrid(np.arange(-10, 10, 0.05), np.arange(-10, 10, 0.05))
x = XX.reshape((XX.size, 1))
y = YY.reshape((XX.size, 1))
xy_matrix = np.append(x, y, axis=1)
mog = p_lambda1 * gaussian(xy_matrix, mu1, sig1)[:, 0] + \
    p_lambda2 * gaussian(xy_matrix, mu2, sig2)[:, 0]
mog = mog.reshape((XX.shape[0], XX.shape[1]))

plt.figure("EM MoG")
plt.subplot(212)
plt.scatter(X[:, 0], X[:, 1], marker=".")
plt.axis([-6, 6, -1, 9])
plt.contour(XX, YY, mog)
plt.show()
