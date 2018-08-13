import numpy as np
from scipy import optimize
import fitting


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gaussian(X, mu, sig):
    k = 1 / ((2 * np.pi) ** (mu.size / 2) * np.sqrt(np.linalg.det(sig)))
    sig_inv = np.linalg.pinv(sig)
    exp_factors = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        exp_factors[i] = np.exp(-0.5 * np.dot(np.dot((X[i] - mu),
                                                     sig_inv), np.transpose([X[i] - mu])))
    return k * exp_factors


def basic_generative(x_train, x_test, K):
    """Classification based on multivariate measurement vector.

    Input:  x_train   - training data, the last column contains class assignments.
            x_test    - test data, there is no column for class assignments.
            K         - the number of classes, assumed to be represented by 0:K.
    Output: p_lambda  - categorical prior with K parameters.
            mu        - each row is the mean of the training examples in class k.
            sig       - covariance matrices, sig[k] is the convariance matrix for the class k.
            posterior - each row is a posterior probability distribution over the K classes.
    """
    I = x_train.shape[0]
    dimensionality = x_train.shape[1] - 1
    x_train_per_class = [[] for x in range(K)]
    class_counts = np.zeros(K, dtype=int)
    for i in range(I):
        k = int(x_train[i, -1])
        current_row = x_train[i, :-1]
        x_train_per_class[k].append(current_row)
        class_counts[k] += 1
    mu = np.zeros((K, dimensionality))
    sig = [np.zeros((dimensionality, dimensionality)) for x in range(K)]
    p_lambda = np.zeros((K, 1))
    for k in range(K):
        if class_counts[k] == 0:
            continue

        mu[k] = np.sum(x_train_per_class[k], axis=0) / class_counts[k]

        for i in range(class_counts[k]):
            mat = x_train_per_class[k][i] - mu[k]
            mat = np.dot(np.transpose([mat]), mat)
            sig[k] += mat
        sig[k] /= class_counts[k]

        p_lambda[k] = class_counts[k] / I

    likelihoods = np.zeros((x_test.shape[0], K))
    for k in range(K):
        likelihoods[:, k] = gaussian(x_test, mu[k], sig[k])[:, 0]

    denominator = 1 / np.dot(likelihoods, p_lambda)
    posterior = np.dot(likelihoods, np.diagflat(p_lambda))

    if posterior.ndim == 1:
        posterior_t = np.transpose([posterior])
    else:
        posterior_t = np.transpose(posterior)

    posterior = np.dot(posterior_t, np.diagflat(denominator))

    if posterior.ndim == 1:
        posterior = np.transpose([posterior])
    else:
        posterior = np.transpose(posterior)

    for k in range(K):
        if class_counts[k] == 0:
            posterior[:, k] = 0

    return (p_lambda, mu, sig, posterior)


def fit_logistic(X, w, var_prior, X_test, initial_phi):
    """MAP logistic regression.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            var_prior   - scale factor for the prior spherical covariance.
            X_test      - test examples for which we need to make predictions.
            initial_phi - (D + 1) * 1 vector that represents the initial solution.
    Output: predictions - 1 * I_test row vector containing the predicted class values for
                          the input data in X_test.
            phi         - D + 1 row vector containing the coefficients for the
                          linear activation function.
    """
    # Find the MAP estimate of the parameters phi
    phi = optimize.minimize(
        _fit_logr_cost,
        initial_phi.reshape(initial_phi.size),
        args=(X, w, var_prior),
        method="Newton-CG",
        jac=_fit_logr_jac,
        hess=_fit_logr_hess
    ).x
    predictions = sigmoid(phi @ X_test)
    return (predictions, phi)


def _fit_logr_cost(phi, X, w, var_prior):
    I = X.shape[1]
    D = X.shape[0] - 1
    L = I * (-np.log(fitting.gaussian_pdf(
        phi.reshape((1, phi.size)),
        np.zeros(D + 1),
        var_prior * np.eye(D + 1)
    )[0, 0]))

    predictions = sigmoid(phi.reshape((1, D + 1)) @ X)
    for i in range(I):
        y = predictions[0, i]
        if w[i, 0] == 1:
            L -= np.log(y)
        else:
            L -= np.log(1 - y)
    return L


def _fit_logr_jac(phi, X, w, var_prior):
    I = X.shape[1]
    D = X.shape[0] - 1
    g = I * phi / var_prior
    predictions = sigmoid(phi.reshape((1, D + 1)) @ X)
    for i in range(I):
        y = predictions[0, i]
        g += (y - w[i, 0]) * X[:, i]
    return g


def _fit_logr_hess(phi, X, w, var_prior):
    I = X.shape[1]
    D = X.shape[0] - 1
    H = I * (1 / var_prior) * np.ones((D + 1, D + 1))
    predictions = sigmoid(phi.reshape((1, D + 1)) @ X)
    for i in range(I):
        y = predictions[0, i]
        x_i = X[:, i].reshape((D + 1, 1))
        H += y * (1 - y) * (x_i @ x_i.transpose())
    return H


def fit_by_logistic(X, w, var_prior, X_test, initial_phi):
    """Bayesian logistic regression.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            var_prior   - scale factor for the prior spherical covariance.
            X_test      - test examples for which we need to make predictions.
            initial_phi - (D + 1) * 1 vector that represents the initial solution.
    Output: predictions - 1 * I_test row vector containing the predicted class values for
                          the input data in X_test.
            phi         - D + 1 row vector containing the coefficients for the
                          linear activation function.
    """
    I_test = X_test.shape[1]
    D = X.shape[0] - 1

    # Find the MAP estimate of the parameters phi
    phi = optimize.minimize(
        _fit_logr_cost,
        initial_phi.reshape(initial_phi.size),
        args=(X, w, var_prior),
        method="Newton-CG",
        jac=_fit_logr_jac,
        hess=_fit_logr_hess
    ).x

    # Compute the Hessian at phi
    H = _fit_logr_hess(phi, X, w, var_prior)

    mu = phi
    var = -np.linalg.pinv(H)

    mu_a = mu.reshape((1, D + 1)) @ X_test
    var_a_temp = X_test.transpose() @ var
    var_a = np.zeros((1, I_test))
    for i in range(I_test):
        var_a[0, i] = var_a_temp[i, :] @ X_test[:, i]

    p_lambda = sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
    predictions = p_lambda
    return (predictions, phi)
