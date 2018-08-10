import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.abspath(__file__), '..', '..')
    )
)

import fitting
import kernel
import regression
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sig):
    return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-0.5 * (x - mu) ** 2 / (sig ** 2))


argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_regression.py fit_linear|fit_by_linear|fit_gaussian_process"
          "|fit_sparse_linear|fit_dual_gaussian_process|fit_relevance_vector")
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

elif argv[1] == "fit_gaussian_process":
    granularity = 500
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)

    # Generate 2D data from normal distribution
    mu = np.array([[-3, 2], [0, -3], [4, 3]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    X_data = np.append(
        np.append(
            np.random.multivariate_normal(mu[0], sig, 5),
            np.random.multivariate_normal(mu[1], sig, 5),
            axis=0
        ),
        np.random.multivariate_normal(mu[2], sig, 5),
        axis=0
    )

    # Prepare the training input
    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data[:, 0].reshape((1, X_data.shape[0])),
        axis=0
    )
    w = X_data[:, 1].reshape((X_data.shape[0], 1))
    var_prior = 6
    X_test = domain.reshape((1, granularity))
    X_test = np.append(np.ones((1, granularity)), X_test, axis=0)

    # Train 6 Gaussian process regression model for different values for nu
    plt.figure("Fit Gaussian process regression")
    nus = [0.5, 0.7, 0.9, 1.5, 2, 3]
    for nu_index, nu in enumerate(nus):
        mu_test, var_test = regression.fit_gaussian_process(
            X_train, w, var_prior, X_test,
            lambda x_i, x_j: kernel.gaussian(x_i, x_j, nu)
        )
        Z = np.zeros((granularity, granularity))
        for j in range(granularity):
            mu = mu_test[j, 0]
            var = var_test[j, 0]
            for i in range(granularity):
                ww = domain[i]
                Z[i, j] = gaussian(ww, mu, var)

        plt.subplot(3, 2, nu_index + 1)
        plt.pcolor(X, Y, Z)
        plt.scatter(X_data[:, 0], X_data[:, 1], edgecolors="w")
        plt.axis([-5, 5, -5, 5])

    plt.show()

elif argv[1] == 'fit_sparse_linear':
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((X.size, 1))
    y = Y.reshape((Y.size, 1))

    # Generate data
    xy_matrix = np.append(
        np.append(np.ones((X.size, 1)), x, axis=1),
        y, axis=1
    )
    phi = np.array([[0], [0.8], [5]])  # plane equation cofficients
    Z = xy_matrix @ phi

    # Offset the points a bit from the plane
    offset = 0.8 * np.random.standard_normal(Z.size)
    X_points = x + offset
    Y_points = y + offset
    Z_points = Z + offset

    # Remove points near the borders
    selector_1 = np.logical_and(X_points < 4.5, X_points > -4.5)
    selector_2 = np.logical_and(Y_points < 4.5, Y_points > -4.5)
    selector = np.logical_and(selector_1, selector_2)
    X_points = X_points[selector]
    Y_points = Y_points[selector]
    Z_points = Z_points[selector]

    # Decrease the number of points
    selector = np.random.permutation(X_points.size)
    selector = selector[0:30]
    X_points = X_points[selector]
    Y_points = Y_points[selector]
    Z_points = Z_points[selector]

    # Prepare the training data
    I = X_points.size
    X_train = np.append(np.ones((1, I)), [X_points], axis=0)
    X_train = np.append(X_train, [Y_points], axis=0)
    w = Z_points.reshape((Z_points.size, 1))
    X_test = np.append(
        np.ones((1, granularity * granularity)),
        x.transpose(), axis=0
    )
    X_test = np.append(X_test, y.transpose(), axis=0)

    # Fit Bayesian linear regression model
    var_prior = 6
    mu_test, var_test, var, A_inv = regression.fit_by_linear(
        X_train, w, var_prior, X_test
    )
    ZZ = mu_test.reshape((granularity, granularity))
    plt.figure("Sparse linear regression")
    plt.subplot(1, 2, 1)
    plt.pcolor(X, Y, ZZ)
    plt.scatter(X_points, Y_points, 40, Z_points, edgecolors="w")

    # Fit sparse linear regression model
    nu = 0.0005
    mu_test, var_test = regression.fit_sparse_linear(
        X_train, w, nu, X_test
    )
    ZZ = mu_test.reshape((granularity, granularity))
    plt.subplot(1, 2, 2)
    plt.pcolor(X, Y, ZZ)
    plt.scatter(X_points, Y_points, 40, Z_points, edgecolors="w")

    plt.show()

elif argv[1] == 'fit_dual_gaussian_process':
    I = 70
    D = 100
    I_test = 10

    X_train = np.random.standard_normal((D, I))
    X_train = np.append(np.ones((1, I)), X_train, axis=0)

    # Evaluate a plane equation in D-dimensional space on the training data points.
    phi = np.ones((D + 1, 1))  # plane equation coefficients
    w = X_train.transpose() @ phi

    X_test = np.random.standard_normal((D, I_test))
    X_test = np.append(np.ones((1, I_test)), X_test, axis=0)

    var_prior = 6
    mu_test, var_test = regression.fit_dual_gaussian_process(
        X_train, w, var_prior, X_test, kernel.linear
    )

    original_model_predictions = X_test.transpose() @ phi
    learned_model_predictions = mu_test
    print(np.append(original_model_predictions, learned_model_predictions, axis=1))

elif argv[1] == 'fit_relevance_vector':
    granularity = 1000
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)

    # Generate 2D data from normal distribution
    mu = np.array([[-3, 2], [0, -3], [4, 3]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    X_data = np.append(
        np.append(
            np.random.multivariate_normal(mu[0], sig, 10),
            np.random.multivariate_normal(mu[1], sig, 10),
            axis=0
        ),
        np.random.multivariate_normal(mu[2], sig, 10),
        axis=0
    )

    # Prepare training input
    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data[:, 0].reshape((1, X_data.shape[0])),
        axis=0
    )
    w = X_data[:, 1].reshape((X_data.shape[0], 1))
    nu = 0.0005
    X_test = np.append(
        np.ones((1, granularity)),
        domain.reshape((1, granularity)),
        axis=0
    )

    # Fit a relevance vector regression model
    mu_test, var_test, relevant_points = regression.fit_relevance_vector(
        X_train, w, nu, X_test,
        lambda x_i, x_j: kernel.gaussian(x_i, x_j, 2)
    )

    plt.figure("Relevance vector regression")
    Z = np.zeros((granularity, granularity))
    for j in range(granularity):
        mu = mu_test[j, 0]
        var = var_test[j, 0]
        for i in range(granularity):
            ww = domain[i]
            Z[i, j] = gaussian(ww, mu, var)
    plt.pcolor(X, Y, Z)

    # Plot the non-relevant data points
    relevant_points = relevant_points.reshape(relevant_points.size)
    plt.scatter(
        X_data[np.logical_not(relevant_points), 0],
        X_data[np.logical_not(relevant_points), 1],
        50, edgecolors="w"
    )

    # Plot the relevant data points
    plt.scatter(
        X_data[relevant_points, 0],
        X_data[relevant_points, 1],
        200, edgecolors="w"
    )

    plt.axis([-5, 5, -5, 5])
    plt.show()

else:
    print("Usage: python test_regression.py fit_linear|fit_by_linear|fit_gaussian_process"
          "|fit_sparse_linear|fit_dual_gaussian_process|fit_relevance_vector")
