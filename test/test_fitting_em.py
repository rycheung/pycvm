import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import fitting
import numpy as np
import matplotlib.pyplot as plt

argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_fitting_em.py em_mog|em_t_distribution")
    sys.exit(-1)

if argv[1] == 'em_mog':
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

elif argv[1] == 'em_t_distribution':
    N = 1000
    data_mu = np.array([1.0, 2.0])
    data_sig = np.array([[2.0, 0], [0, 0.5]])
    x = np.random.multivariate_normal(data_mu, data_sig, N)

    N_outliers = 20
    outliers_mu = np.array([-4, 7])
    outliers_sig = np.array([[0.2, 0], [0, 0.2]])
    outliers = np.random.multivariate_normal(outliers_mu, outliers_sig, N_outliers)

    x_plus_outliers = np.append(x, outliers, axis=0)
    # Fit a Gaussian to the original data and to the data with outliers
    p_lambda1, mu1, sig1 = fitting.em_mog(x, 1, 0.01)
    p_lambda2, mu2, sig2 = fitting.em_mog(x_plus_outliers, 1, 0.01)

    # Fit a t-distribution to the data with outliers
    t_mu, t_sig, t_nu = fitting.em_t_distribution(x_plus_outliers, 0.01)

    XX, YY = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))
    xy_matrix = np.append(XX.reshape((XX.size, 1)), YY.reshape((XX.size, 1)), axis=1)

    plt.figure("EM t-distribution")

    plt.subplot(131)
    plt.scatter(x[:, 0], x[:, 1], marker=".")
    temp = fitting.gaussian_pdf(xy_matrix, mu1[0], sig1[0])[:, 0]
    gaussian1 = temp.reshape((XX.shape[0], XX.shape[1]))
    plt.contour(XX, YY, gaussian1)
    plt.axis([-6, 6, -1, 8])
    
    plt.subplot(132)
    plt.scatter(x_plus_outliers[:, 0], x_plus_outliers[:, 1], marker=".")
    temp = fitting.gaussian_pdf(xy_matrix, mu2[0], sig2[0])[:, 0]
    gaussian2 = temp.reshape((XX.shape[0], XX.shape[1]))
    plt.contour(XX, YY, gaussian2)
    plt.axis([-6, 6, -1, 8])

    plt.subplot(133)
    plt.scatter(x_plus_outliers[:, 0], x_plus_outliers[:, 1], marker=".")
    temp = fitting.mul_t_pdf(xy_matrix, t_mu, t_sig, t_nu)
    t_distribution = temp.reshape((XX.shape[0], XX.shape[1]))
    plt.contour(XX, YY, t_distribution)
    plt.axis([-6, 6, -1, 8])

    plt.show()
