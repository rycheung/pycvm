import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import fitting
import numpy as np
import matplotlib.pyplot as plt

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
    mog[:, 0] += p_lambda[k] * \
        fitting.gaussian_pdf(xy_matrix, mu[k], sig[k])[:, 0]
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
mog = p_lambda1 * fitting.gaussian_pdf(xy_matrix, mu1, sig1)[:, 0] + \
    p_lambda2 * fitting.gaussian_pdf(xy_matrix, mu2, sig2)[:, 0]
mog = mog.reshape((XX.shape[0], XX.shape[1]))

plt.figure("EM MoG")
plt.subplot(212)
plt.scatter(X[:, 0], X[:, 1], marker=".")
plt.axis([-6, 6, -1, 9])
plt.contour(XX, YY, mog)
plt.show()
