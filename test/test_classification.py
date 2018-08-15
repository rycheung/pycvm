import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.abspath(__file__), "..", "..")
    )
)

import kernel
import classification
import numpy as np
import matplotlib.pyplot as plt

argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_classification.py fit_logistic|fit_by_logistic"
          "|fit_dual_logistic|fit_dual_by_logistic|fit_gaussian_process"
          "|fit_relevance_vector|fit_incremental_logistic")
    sys.exit(-1)

if argv[1] == "fit_logistic":
    plt.figure("MAP logistic regression")
    # 1D logistic regression
    plt.subplot(1, 2, 1)
    I_0 = 10
    mu_0 = -2
    sig_0 = 1.5
    class_0 = np.random.normal(mu_0, sig_0, (1, I_0))
    plt.scatter(class_0, np.zeros((1, I_0)), 50, c="r", edgecolors="k")

    I_1 = 10
    mu_1 = 2
    sig_1 = 1.5
    class_1 = np.random.normal(mu_1, sig_1, (1, I_1))
    plt.scatter(class_1, np.zeros((1, I_1)), 50, c="g", edgecolors="k")

    # Prepare training data
    X = np.append(class_0, class_1, axis=1)
    X = np.append(np.ones((1, I_0 + I_1)), X, axis=0)
    w = np.append(np.zeros((I_0, 1)), np.ones((I_1, 1)), axis=0)
    var_prior = 6
    X_test = np.arange(-5, 5, 0.1)
    X_test = np.append(
        np.ones((1, X_test.size)),
        X_test.reshape(1, X_test.size),
        axis=0
    )

    # Fit a logistic regression model
    initial_phi = np.array([[1], [1]])
    predictions, phi = classification.fit_logistic(
        X, w, var_prior, X_test, initial_phi
    )
    plt.plot(np.arange(-5, 5, 0.1), predictions)
    decision_boundary = -phi[0] / phi[1]
    plt.plot(
        decision_boundary * np.ones(10),
        np.linspace(-2, 2, 10)
    )

    plt.axis([-5, 5, -0.1, 1.1])

    # 2D logistic regression
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((1, X.size))
    y = Y.reshape((1, Y.size))

    mu = np.array([[-1, 2.5], [1, -2.5]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    points_per_class = 20
    X_data = np.append(
        np.random.multivariate_normal(mu[0], sig, points_per_class),
        np.random.multivariate_normal(mu[1], sig, points_per_class),
        axis=0
    )

    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data.transpose(),
        axis=0
    )
    w = np.append(
        np.zeros((points_per_class, 1)),
        np.ones((points_per_class, 1)),
        axis=0
    )
    var_prior = 6
    X_test = np.append(np.ones((1, granularity * granularity)), x, axis=0)
    X_test = np.append(X_test, y, axis=0)
    initial_phi = np.array([[1], [1], [1]])
    predictions, phi = classification.fit_logistic(
        X_train, w, var_prior, X_test, initial_phi
    )

    plt.subplot(1, 2, 2)
    Z = predictions.reshape(granularity, granularity)
    plt.pcolor(X, Y, Z)

    selector = np.arange(points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="r", edgecolors="w")

    selector = np.arange(points_per_class, 2 * points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="g", edgecolors="w")

    decision_boundary = -(phi[0] + phi[1] * domain) / phi[2]
    plt.plot(domain, decision_boundary, "r")

    plt.axis([a, b, a, b])

    plt.show()

elif argv[1] == "fit_by_logistic":
    plt.figure("Bayesian logistic regression")
    # 1D Bayesian logistic regression
    plt.subplot(1, 2, 1)
    I_0 = 10
    mu_0 = -2
    sig_0 = 1.5
    class_0 = np.random.normal(mu_0, sig_0, (1, I_0))
    plt.scatter(class_0, np.zeros((1, I_0)), 50, c="r", edgecolors="k")

    I_1 = 10
    mu_1 = 2
    sig_1 = 1.5
    class_1 = np.random.normal(mu_1, sig_1, (1, I_1))
    plt.scatter(class_1, np.zeros((1, I_1)), 50, c="g", edgecolors="k")

    # Prepare training data
    X = np.append(class_0, class_1, axis=1)
    X = np.append(np.ones((1, I_0 + I_1)), X, axis=0)
    w = np.append(np.zeros((I_0, 1)), np.ones((I_1, 1)), axis=0)
    var_prior = 6
    X_test = np.arange(-5, 5, 0.1)
    X_test = np.append(
        np.ones((1, X_test.size)),
        X_test.reshape(1, X_test.size),
        axis=0
    )

    # Fit a logistic regression model
    initial_phi = np.array([[1], [1]])
    predictions, phi = classification.fit_by_logistic(
        X, w, var_prior, X_test, initial_phi
    )
    plt.plot(np.arange(-5, 5, 0.1), predictions)
    decision_boundary = -phi[0] / phi[1]
    plt.plot(
        decision_boundary * np.ones(10),
        np.linspace(-2, 2, 10)
    )

    plt.axis([-5, 5, -0.1, 1.1])

    # 2D Bayesian logistic regression
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((1, X.size))
    y = Y.reshape((1, Y.size))

    mu = np.array([[-1, 2.5], [1, -2.5]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    points_per_class = 20
    X_data = np.append(
        np.random.multivariate_normal(mu[0], sig, points_per_class),
        np.random.multivariate_normal(mu[1], sig, points_per_class),
        axis=0
    )

    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data.transpose(),
        axis=0
    )
    w = np.append(
        np.zeros((points_per_class, 1)),
        np.ones((points_per_class, 1)),
        axis=0
    )
    var_prior = 6
    X_test = np.append(np.ones((1, granularity * granularity)), x, axis=0)
    X_test = np.append(X_test, y, axis=0)
    initial_phi = np.array([[1], [1], [1]])
    predictions, phi = classification.fit_by_logistic(
        X_train, w, var_prior, X_test, initial_phi
    )

    plt.subplot(1, 2, 2)
    Z = predictions.reshape(granularity, granularity)
    plt.pcolor(X, Y, Z)

    selector = np.arange(points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="r", edgecolors="w")

    selector = np.arange(points_per_class, 2 * points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="g", edgecolors="w")

    decision_boundary = -(phi[0] + phi[1] * domain) / phi[2]
    plt.plot(domain, decision_boundary, "r")

    plt.axis([a, b, a, b])

    plt.show()

elif argv[1] == "fit_dual_logistic":
    plt.figure("Dual logistic regression")
    # 1D dual logistic regression
    plt.subplot(1, 2, 1)
    I_0 = 10
    mu_0 = -2
    sig_0 = 1.5
    class_0 = np.random.normal(mu_0, sig_0, (1, I_0))
    plt.scatter(class_0, np.zeros((1, I_0)), 50, c="r", edgecolors="k")

    I_1 = 10
    mu_1 = 2
    sig_1 = 1.5
    class_1 = np.random.normal(mu_1, sig_1, (1, I_1))
    plt.scatter(class_1, np.zeros((1, I_1)), 50, c="g", edgecolors="k")

    # Prepare training data
    X = np.append(class_0, class_1, axis=1)
    X = np.append(np.ones((1, I_0 + I_1)), X, axis=0)
    w = np.append(np.zeros((I_0, 1)), np.ones((I_1, 1)), axis=0)
    var_prior = 6
    X_test = np.arange(-5, 5, 0.1)
    X_test = np.append(
        np.ones((1, X_test.size)),
        X_test.reshape(1, X_test.size),
        axis=0
    )

    initial_psi = np.zeros((X.shape[1], 1))
    predictions, psi = classification.fit_dual_logistic(
        X, w, var_prior, X_test, initial_psi
    )
    phi = X @ psi.reshape((psi.size, 1))

    plt.plot(np.arange(-5, 5, 0.1), predictions)
    decision_boundary = -phi[0, 0] / phi[1, 0]
    plt.plot(
        decision_boundary * np.ones(10),
        np.linspace(-2, 2, 10)
    )

    plt.axis([-5, 5, -0.1, 1.1])

    # 2D dual logistic regression
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((1, X.size))
    y = Y.reshape((1, Y.size))

    mu = np.array([[-1, 2.5], [1, -2.5]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    points_per_class = 20
    X_data = np.append(
        np.random.multivariate_normal(mu[0], sig, points_per_class),
        np.random.multivariate_normal(mu[1], sig, points_per_class),
        axis=0
    )

    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data.transpose(),
        axis=0
    )
    w = np.append(
        np.zeros((points_per_class, 1)),
        np.ones((points_per_class, 1)),
        axis=0
    )
    var_prior = 6
    X_test = np.append(np.ones((1, granularity * granularity)), x, axis=0)
    X_test = np.append(X_test, y, axis=0)
    initial_psi = np.zeros((X_train.shape[1], 1))
    predictions, psi = classification.fit_dual_logistic(
        X_train, w, var_prior, X_test, initial_psi
    )
    phi = X_train @ psi.reshape((psi.size, 1))

    plt.subplot(1, 2, 2)
    Z = predictions.reshape(granularity, granularity)
    plt.pcolor(X, Y, Z)

    selector = np.arange(points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="r", edgecolors="w")

    selector = np.arange(points_per_class, 2 * points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="g", edgecolors="w")

    decision_boundary = -(phi[0, 0] + phi[1, 0] * domain) / phi[2, 0]
    plt.plot(domain, decision_boundary, "r")

    plt.axis([a, b, a, b])

    plt.show()

elif argv[1] == "fit_dual_by_logistic":
    plt.figure("Dual Bayesian logistic regression")
    # 1D dual Bayesian logistic regression
    plt.subplot(1, 2, 1)
    I_0 = 10
    mu_0 = -2
    sig_0 = 1.5
    class_0 = np.random.normal(mu_0, sig_0, (1, I_0))
    plt.scatter(class_0, np.zeros((1, I_0)), 50, c="r", edgecolors="k")

    I_1 = 10
    mu_1 = 2
    sig_1 = 1.5
    class_1 = np.random.normal(mu_1, sig_1, (1, I_1))
    plt.scatter(class_1, np.zeros((1, I_1)), 50, c="g", edgecolors="k")

    # Prepare training data
    X = np.append(class_0, class_1, axis=1)
    X = np.append(np.ones((1, I_0 + I_1)), X, axis=0)
    w = np.append(np.zeros((I_0, 1)), np.ones((I_1, 1)), axis=0)
    var_prior = 6
    X_test = np.arange(-5, 5, 0.1)
    X_test = np.append(
        np.ones((1, X_test.size)),
        X_test.reshape(1, X_test.size),
        axis=0
    )

    initial_psi = np.zeros((X.shape[1], 1))
    predictions, psi = classification.fit_dual_by_logistic(
        X, w, var_prior, X_test, initial_psi
    )
    phi = X @ psi.reshape((psi.size, 1))

    plt.plot(np.arange(-5, 5, 0.1), predictions)
    decision_boundary = -phi[0, 0] / phi[1, 0]
    plt.plot(
        decision_boundary * np.ones(10),
        np.linspace(-2, 2, 10)
    )

    plt.axis([-5, 5, -0.1, 1.1])

    # 2D dual logistic regression
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((1, X.size))
    y = Y.reshape((1, Y.size))

    mu = np.array([[-1, 2.5], [1, -2.5]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    points_per_class = 20
    X_data = np.append(
        np.random.multivariate_normal(mu[0], sig, points_per_class),
        np.random.multivariate_normal(mu[1], sig, points_per_class),
        axis=0
    )

    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data.transpose(),
        axis=0
    )
    w = np.append(
        np.zeros((points_per_class, 1)),
        np.ones((points_per_class, 1)),
        axis=0
    )
    var_prior = 6
    X_test = np.append(np.ones((1, granularity * granularity)), x, axis=0)
    X_test = np.append(X_test, y, axis=0)
    initial_psi = np.zeros((X_train.shape[1], 1))
    predictions, psi = classification.fit_dual_by_logistic(
        X_train, w, var_prior, X_test, initial_psi
    )
    phi = X_train @ psi.reshape((psi.size, 1))

    plt.subplot(1, 2, 2)
    Z = predictions.reshape(granularity, granularity)
    plt.pcolor(X, Y, Z)

    selector = np.arange(points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="r", edgecolors="w")

    selector = np.arange(points_per_class, 2 * points_per_class)
    plt.scatter(X_data[selector, 0], X_data[selector, 1],
                50, c="g", edgecolors="w")

    decision_boundary = -(phi[0, 0] + phi[1, 0] * domain) / phi[2, 0]
    plt.plot(domain, decision_boundary, "r")

    plt.axis([a, b, a, b])

    plt.show()

elif argv[1] == "fit_gaussian_process":
    plt.figure("1D Gaussian process classification")
    I_0 = 20
    mu_0 = -2
    sig_0 = 1.5
    class_0 = np.random.normal(mu_0, sig_0, (1, I_0))

    I_1 = 20
    mu_1 = 2
    sig_1 = 1.5
    class_1 = np.random.normal(mu_1, sig_1, (1, I_1))

    # Prepare training data
    X = np.append(class_0, class_1, axis=1)
    X = np.append(np.ones((1, I_0 + I_1)), X, axis=0)
    w = np.append(np.zeros((I_0, 1)), np.ones((I_1, 1)), axis=0)
    var_prior = 6
    X_test = np.arange(-5, 5, 0.1)
    X_test = np.append(
        np.ones((1, X_test.size)),
        X_test.reshape(1, X_test.size),
        axis=0
    )

    p_lambdas = [0.3, 1, 5, 15]
    for index, p_lambda in enumerate(p_lambdas):
        initial_psi = np.zeros((X.shape[1], 1))
        predictions, psi = classification.fit_gaussian_process(
            X, w, var_prior, X_test, initial_psi,
            lambda x_i, x_j: kernel.gaussian(x_i, x_j, p_lambda)
        )

        plt.subplot(2, 2, index + 1)
        plt.plot(np.arange(-5, 5, 0.1), predictions)
        plt.scatter(class_0, np.zeros((1, I_0)), 50, c="r", edgecolors="k")
        plt.scatter(class_1, np.zeros((1, I_1)), 50, c="g", edgecolors="k")
        plt.axis([-5, 5, -0.1, 1.1])

    plt.figure("2D Gaussian process classification")
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((1, X.size))
    y = Y.reshape((1, Y.size))

    mu = np.array([[-1, 2.5], [1, -2.5]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    points_per_class = 20
    X_data = np.append(
        np.random.multivariate_normal(mu[0], sig, points_per_class),
        np.random.multivariate_normal(mu[1], sig, points_per_class),
        axis=0
    )

    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data.transpose(),
        axis=0
    )
    w = np.append(
        np.zeros((points_per_class, 1)),
        np.ones((points_per_class, 1)),
        axis=0
    )
    var_prior = 6
    X_test = np.append(np.ones((1, granularity * granularity)), x, axis=0)
    X_test = np.append(X_test, y, axis=0)

    p_lambdas = [0.3, 1, 5, 15]
    for index, p_lambda in enumerate(p_lambdas):
        initial_psi = np.zeros((X_train.shape[1], 1))
        predictions, psi = classification.fit_gaussian_process(
            X_train, w, var_prior, X_test, initial_psi,
            lambda x_i, x_j: kernel.gaussian(x_i, x_j, p_lambda)
        )

        plt.subplot(2, 2, index + 1)
        Z = predictions.reshape(granularity, granularity)
        plt.pcolor(X, Y, Z)

        selector = np.arange(points_per_class)
        plt.scatter(X_data[selector, 0], X_data[selector, 1],
                    50, c="r", edgecolors="w")

        selector = np.arange(points_per_class, 2 * points_per_class)
        plt.scatter(X_data[selector, 0], X_data[selector, 1],
                    50, c="g", edgecolors="w")

        plt.axis([a, b, a, b])

    plt.show()

elif argv[1] == "fit_relevance_vector":
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((1, X.size))
    y = Y.reshape((1, Y.size))

    mu = np.array([[-1, 2.5], [1, -2.5]])
    sig = np.array([[0.5, 0], [0, 0.5]])
    points_per_class = 20
    X_data = np.append(
        np.random.multivariate_normal(mu[0], sig, points_per_class),
        np.random.multivariate_normal(mu[1], sig, points_per_class),
        axis=0
    )

    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data.transpose(),
        axis=0
    )
    w = np.append(
        np.zeros((points_per_class, 1)),
        np.ones((points_per_class, 1)),
        axis=0
    )
    X_test = np.append(np.ones((1, granularity * granularity)), x, axis=0)
    X_test = np.append(X_test, y, axis=0)

    p_lambdas = [0.3, 1, 5, 15]
    plt.figure("Relevance vector classification")
    for index, p_lambda in enumerate(p_lambdas):
        initial_psi = np.zeros((X_train.shape[1], 1))
        nu = 0.0005
        predictions, relevant_points = classification.fit_relevance_vector(
            X_train, w, nu, X_test, initial_psi,
            lambda x_i, x_j: kernel.gaussian(x_i, x_j, p_lambda)
        )

        plt.subplot(2, 2, index + 1)
        Z = predictions.reshape(granularity, granularity)
        plt.pcolor(X, Y, Z)

        selector0 = relevant_points.reshape(relevant_points.size).copy()
        selector1 = relevant_points.reshape(relevant_points.size).copy()

        selector0[points_per_class:(2 * points_per_class)] = False
        selector1[0:points_per_class] = False

        selector0n = selector0.copy()
        selector0n[0:points_per_class] = np.logical_not(
            selector0n[0:points_per_class]
        )

        selector1n = selector1.copy()
        selector1n[points_per_class:(2 * points_per_class)] = np.logical_not(
            selector1n[points_per_class:(2 * points_per_class)]
        )

        plt.scatter(X_data[selector0, 0], X_data[selector0, 1],
                    200, c="r", edgecolors="k")
        plt.scatter(X_data[selector0n, 0], X_data[selector0n, 1],
                    50, c="r", edgecolors="k")
        plt.scatter(X_data[selector1, 0], X_data[selector1, 1],
                    200, c="b", edgecolors="k")
        plt.scatter(X_data[selector1n, 0], X_data[selector1n, 1],
                    50, c="b", edgecolors="k")

        plt.axis([a, b, a, b])
    plt.show()

elif argv[1] == "fit_incremental_logistic":
    plt.figure("1D incremental fitting of logistic classification")
    I_0 = 20
    mu_0 = -2
    sig_0 = 1.5
    class_0 = np.random.normal(mu_0, sig_0, (1, I_0))

    I_1 = 20
    mu_1 = 2
    sig_1 = 1.5
    class_1 = np.random.normal(mu_1, sig_1, (1, I_1))

    X = np.append(class_0, class_1, axis=1)
    X = np.append(np.ones((1, I_0 + I_1)), X, axis=0)
    w = np.append(np.zeros((I_0, 1)), np.ones((I_1, 1)), axis=0)
    var_prior = 6
    X_test = np.arange(-5, 5, 0.1)
    X_test = np.append(
        np.ones((1, X_test.size)),
        X_test.reshape(1, X_test.size),
        axis=0
    )

    Ks = [1, 5, 10, 20]
    for index, K in enumerate(Ks):
        plt.subplot(2, 2, index + 1)
        predictions = classification.fit_incremental_logistic(X, w, X_test, K)
        plt.plot(np.arange(-5, 5, 0.1), predictions)
        plt.scatter(class_0, np.zeros((1, I_0)), 50, c="r", edgecolors="k")
        plt.scatter(class_1, np.zeros((1, I_1)), 50, c="g", edgecolors="k")
        plt.axis([-5, 5, -0.1, 1.1])

    plt.figure("2D incremental fitting of logistic classification")
    granularity = 100
    a = -5
    b = 5
    domain = np.linspace(a, b, granularity)
    X, Y = np.meshgrid(domain, domain)
    x = X.reshape((1, X.size))
    y = Y.reshape((1, Y.size))

    mu = np.array([[-1, 1], [1, -1]])
    sig = np.array([[2, 0], [0, 2]])
    points_per_class = 20
    X_data = np.append(
        np.random.multivariate_normal(mu[0], sig, points_per_class),
        np.random.multivariate_normal(mu[1], sig, points_per_class),
        axis=0
    )
    X_train = np.append(
        np.ones((1, X_data.shape[0])),
        X_data.transpose(),
        axis=0
    )
    w = np.append(
        np.zeros((points_per_class, 1)),
        np.ones((points_per_class, 1)),
        axis=0
    )
    X_test = np.append(np.ones((1, granularity * granularity)), x, axis=0)
    X_test = np.append(X_test, y, axis=0)

    Ks = [1, 5, 10, 20]
    for index, K in enumerate(Ks):
        predictions = classification.fit_incremental_logistic(
            X_train, w, X_test, K)

        plt.subplot(2, 2, index + 1)
        Z = predictions.reshape(granularity, granularity)
        plt.pcolor(X, Y, Z)

        selector = np.arange(points_per_class)
        plt.scatter(X_data[selector, 0], X_data[selector, 1],
                    50, c="r", edgecolors="w")

        selector = np.arange(points_per_class, 2 * points_per_class)
        plt.scatter(X_data[selector, 0], X_data[selector, 1],
                    50, c="g", edgecolors="w")

        plt.axis([a, b, a, b])

    plt.show()

else:
    print("Usage: python test_classification.py fit_logistic|fit_by_logistic"
          "|fit_dual_logistic|fit_dual_by_logistic|fit_gaussian_process"
          "|fit_relevance_vector|fit_incremental_logistic")
