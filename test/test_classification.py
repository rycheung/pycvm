import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.abspath(__file__), "..", "..")
    )
)

import classification
import numpy as np
import matplotlib.pyplot as plt

argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_classification.py fit_logistic|fit_by_logistic")
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

else:
    print("Usage: python test_classification.py fit_logistic|fit_by_logistic")
