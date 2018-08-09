import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.abspath(__file__), '..', '..')
    )
)

import fitting
import regression
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sig):
    return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-0.5 * (x - mu) ** 2 / (sig ** 2))


argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_regression.py fit_linear|fit_by_linear")
    sys.exit(-1)

if argv[1] == 'fit_linear':
    I = 50
    x = np.linspace(1, 9, I)
    phi = np.array([[7], [-0.5]])
    sig = 0.6
    w_test = np.zeros((I, 1))
    X = np.append(np.ones((I, 1)), x.reshape((I, 1)), axis=1)
    X_phi = X @ phi

    w = np.random.normal(X_phi, sig)

    fit_phi, fit_sig = regression.fit_linear(X.transpose(), w)
    granularity = 500
    domain = np.linspace(0, 10, granularity)
    X, Y = np.meshgrid(domain, domain)
    XX = np.append(np.ones((granularity, 1)),
                   domain.reshape((granularity, 1)), axis=1)
    temp = XX @ fit_phi
    Z = np.zeros((granularity, granularity))
    for j in range(granularity):
        mu = temp[j, 0]
        for i in range(granularity):
            ww = domain[i]
            Z[i, j] = gaussian(ww, mu, fit_sig)

    plt.figure("Fit linear regression")
    plt.pcolor(X, Y, Z)
    plt.scatter(x, w, edgecolors="w")

    plt.show()

elif argv[1] == "fit_by_linear":
    granularity = 500
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((X.size, 1))
    y = Y.reshape((Y.size, 1))
    xy_matrix = np.append(x, y, axis=1)

    # Compute the prior 2D normal distribution over phi
    mu_1 = np.array([0, 0])
    var_prior = 6
    covariance_1 = var_prior * np.eye(2)
    mvnpdf_1 = fitting.gaussian_pdf(
        xy_matrix, mu_1, covariance_1
    ).reshape(granularity, granularity)

    plt.figure("Prior and posterior over phi")
    plt.subplot(1, 2, 1)
    plt.pcolor(X, Y, mvnpdf_1)

    # Generate the training and test data
    X_train = np.array([[1,  1,  1, 1, 1,   1],
                        [-4, -1, -1, 0, 1, 3.5]])
    w = np.array([4.5, 3, 2, 2.5, 2.5, 0]).reshape((6, 1))
    X_test = domain.reshape((1, granularity))
    X_test = np.append(np.ones((1, granularity)), X_test, axis=0)

    # Fit Bayesian linear regression model
    mu_test, var_test, var, A_inv = regression.fit_by_linear(
        X_train, w, var_prior, X_test
    )

    # Plot the posterior 2D normal distribution over phi
    mu_2 = A_inv @ X_train @ w / var
    mu_2 = mu_2.reshape(mu_2.size)
    covariance_2 = A_inv
    mvnpdf_2 = fitting.gaussian_pdf(
        xy_matrix, mu_2, covariance_2
    ).reshape(granularity, granularity)

    plt.subplot(1, 2, 2)
    plt.pcolor(X, Y, mvnpdf_2)

    # Plot 3 samples from posterior
    phi_samples = np.random.multivariate_normal(mu_2, covariance_2, 3)
    plt.figure("Samples from posterior")
    for k in range(3):
        plt.subplot(3, 1, k + 1)
        XX = domain.reshape((granularity, 1))
        XX = np.append(np.ones((granularity, 1)), XX, axis=1)
        temp = XX @ phi_samples[k, :].reshape(mu_2.size, 1)
        Z = np.zeros((granularity, granularity))
        for j in range(granularity):
            mu = temp[j, 0]
            for i in range(granularity):
                ww = domain[i]
                Z[i, j] = gaussian(ww, mu, var)
        plt.pcolor(X, Y, Z)

    # Plot the main example for Bayesian linear regression
    plt.figure("Fit Bayesian linear regression")
    for j in range(granularity):
        mu = mu_test[j, 0]
        var = var_test[j, 0]
        for i in range(granularity):
            ww = domain[i]
            Z[i, j] = gaussian(ww, mu, var)
    plt.pcolor(X, Y, Z)
    plt.scatter(X_train[1, :], w, edgecolors="w")

    plt.show()

else:
    print("Usage: python test_regression.py fit_linear|fit_by_linear")
