from collections import deque
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
    H = I * (1 / var_prior) * np.eye(D + 1)
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
    H = -_fit_logr_hess(phi, X, w, var_prior)

    mu = phi
    var = -np.linalg.pinv(H)

    mu_a = mu.reshape((1, D + 1)) @ X_test
    var_a_temp = X_test.transpose() @ var
    var_a = np.zeros((1, I_test))
    for i in range(I_test):
        var_a[0, i] = var_a_temp[i, :] @ X_test[:, i]

    p_lambda = sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
    predictions = p_lambda.reshape(I_test)
    return (predictions, phi)


def fit_dual_logistic(X, w, var_prior, X_test, initial_psi):
    """MAP dual logistic regression.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            var_prior   - scale factor for the prior spherical covariance.
            X_test      - test examples for which we need to make predictions.
            initial_psi - I * 1 vector that represents the initial solution.
    Output: predictions - 1 * I_test row vector containing the predicted class values for
                          the input data in X_test.
            psi         - D + 1 row vector containing the coefficients for the
                          activation function.
    """
    I = X.shape[1]
    psi = optimize.minimize(
        _fit_dlogr_cost,
        initial_psi.reshape(initial_psi.size),
        args=(X, w, var_prior),
        method="Newton-CG",
        jac=_fit_dlogr_jac,
        hess=_fit_dlogr_hess
    ).x

    predictions = sigmoid((X @ psi.reshape((I, 1))).transpose() @ X_test)
    predictions = predictions.reshape(predictions.size)
    return (predictions, psi)


def _fit_dlogr_cost(psi, X, w, var_prior):
    I = X.shape[1]
    L = I * (-np.log(fitting.gaussian_pdf(
        psi.reshape((1, psi.size)),
        np.zeros(I),
        var_prior * np.eye(I)
    )[0, 0]))

    predictions = sigmoid((X @ psi.reshape((I, 1))).transpose() @ X)
    for i in range(I):
        y = predictions[0, i]
        if w[i, 0] == 1:
            L -= np.log(y)
        else:
            L -= np.log(1 - y)
    return L


def _fit_dlogr_jac(psi, X, w, var_prior):
    I = X.shape[1]
    D = X.shape[0] - 1
    g = I * psi / var_prior
    g = g.reshape((I, 1))
    predictions = sigmoid((X @ psi.reshape((I, 1))).transpose() @ X)
    for i in range(I):
        y = predictions[0, i]
        g += (y - w[i, 0]) * (X.transpose() @ X[:, i].reshape((D + 1, 1)))
    return g.reshape(I)


def _fit_dlogr_hess(psi, X, w, var_prior):
    I = X.shape[1]
    D = X.shape[0] - 1
    H = I * (1 / var_prior) * np.eye(I)
    predictions = sigmoid((X @ psi.reshape((I, 1))).transpose() @ X)
    for i in range(I):
        y = predictions[0, i]
        temp = X.transpose() @ X[:, i].reshape((D + 1, 1))
        H += y * (1 - y) * (temp @ temp.transpose())
    return H


def fit_dual_by_logistic(X, w, var_prior, X_test, initial_psi):
    """Dual Bayesian logistic regression.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            var_prior   - scale factor for the prior spherical covariance.
            X_test      - test examples for which we need to make predictions.
            initial_psi - I * 1 vector that represents the initial solution.
    Output: predictions - 1 * I_test row vector containing the predicted class values for
                          the input data in X_test.
            psi         - D + 1 row vector containing the coefficients for the
                          activation function.
    """
    I = X.shape[1]
    I_test = X_test.shape[1]
    psi = optimize.minimize(
        _fit_dlogr_cost,
        initial_psi.reshape(initial_psi.size),
        args=(X, w, var_prior),
        method="Newton-CG",
        jac=_fit_dlogr_jac,
        hess=_fit_dlogr_hess
    ).x

    H = -_fit_dlogr_hess(psi, X, w, var_prior)

    mu = psi
    var = -np.linalg.pinv(H)

    mu_a_temp = X.transpose() @ X_test
    mu_a = mu.reshape((1, I)) @ mu_a_temp
    var_a_temp = X @ var @ mu_a_temp
    var_a = np.zeros((1, I_test))
    for i in range(I_test):
        var_a[0, i] = X_test[:, i] @ var_a_temp[:, i]

    p_lambda = sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
    predictions = p_lambda.reshape(I_test)
    return (predictions, psi)


def fit_gaussian_process(X, w, var_prior, X_test, initial_psi, kernel):
    """Gaussian process classification (or kernel logistic regression).

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            var_prior   - scale factor for the prior spherical covariance.
            X_test      - test examples for which we need to make predictions.
            initial_psi - I * 1 vector that represents the initial solution.
            kernel      - the kernel function.
    Output: predictions - 1 * I_test row vector containing the predicted class values for
                          the input data in X_test.
            psi         - D + 1 row vector containing the coefficients for the
                          activation function.
    """
    I = X.shape[1]
    K = np.zeros((I, I))
    for i in range(I):
        for j in range(I):
            K[i, j] = kernel(X[:, i], X[:, j])

    I_test = X_test.shape[1]
    K_test = np.zeros((I, I_test))
    for i in range(I):
        for j in range(I_test):
            K_test[i, j] = kernel(X[:, i], X_test[:, j])

    psi = optimize.minimize(
        _fit_klogr_cost,
        initial_psi.reshape(initial_psi.size),
        args=(X, w, var_prior, K),
        method="Newton-CG",
        jac=_fit_klogr_jac,
        hess=_fit_klogr_hess
    ).x

    H = -_fit_klogr_hess(psi, X, w, var_prior, K)

    mu = psi
    var = -np.linalg.pinv(H)

    mu_a = mu.reshape((1, I)) @ K_test
    var_a_temp = X @ var @ K_test
    var_a = np.zeros((1, I_test))
    for i in range(I_test):
        var_a[0, i] = X_test[:, i] @ var_a_temp[:, i]

    p_lambda = sigmoid(mu_a / np.sqrt(1 + np.pi / 8 * var_a))
    predictions = p_lambda.reshape(I_test)
    return (predictions, psi)


def _fit_klogr_cost(psi, X, w, var_prior, K):
    I = X.shape[1]
    L = I * (-np.log(fitting.gaussian_pdf(
        psi.reshape((1, psi.size)),
        np.zeros(I),
        var_prior * np.eye(I)
    )[0, 0]))

    predictions = sigmoid(psi @ K)
    for i in range(I):
        y = predictions[i]
        if w[i, 0] == 1:
            L -= np.log(y)
        else:
            L -= np.log(1 - y)
    return L


def _fit_klogr_jac(psi, X, w, var_prior, K):
    I = X.shape[1]
    g = I * psi / var_prior
    predictions = sigmoid(psi @ K)
    for i in range(I):
        y = predictions[i]
        g += (y - w[i, 0]) * K[:, i]
    return g


def _fit_klogr_hess(psi, X, w, var_prior, K):
    I = X.shape[1]
    H = I * (1 / var_prior) * np.eye(I)
    predictions = sigmoid(psi @ K)
    for i in range(I):
        y = predictions[i]
        temp = K[:, i].reshape((I, 1))
        H += y * (1 - y) * (temp @ temp.transpose())
    return H


def fit_relevance_vector(X, w, nu, X_test, initial_psi, kernel):
    """Relevance vector classification.

    Input:  X               - (D + 1) * I training data matrix, where D is the dimensionality
                              and I is the number of training examples.
            w               - I * 1 vector containing world states for each example.
            nu              - degrees of freedom.
            X_test          - test examples for which we need to make predictions.
            initial_psi     - I * 1 vector that represents the initial solution.
            kernel          - the kernel function.
    Output: predictions     - 1 * I_test row vector containing the predicted class values for
                              the input data in X_test.
            relevant_points - I * 1 boolean vector where a True at position i indicates that point
                              X[:, i] remained after the elimination phase, i.e. it is relevant.
    """
    I = X.shape[1]
    I_test = X_test.shape[1]
    K = np.zeros((I, I))
    for i in range(I):
        for j in range(I):
            K[i, j] = kernel(X[:, i], X[:, j])

    H = np.ones(I)
    H_old = np.zeros(I)
    iterations_count = 0
    precision = 0.001
    mu = 0
    sig = 0
    while np.sum(np.fabs(H - H_old) > precision) != 0:
        # while iterations_count < 10:
        iterations_count += 1
        H_old = H

        psi = optimize.minimize(
            _fit_rvc_cost,
            initial_psi.reshape(initial_psi.size),
            args=(w, H, K),
            method="Newton-CG",
            jac=_fit_rvc_jac,
            hess=_fit_rvc_hess
        ).x

        # Compute Hessian S at peak
        S = -_fit_rvc_hess(psi, w, H, K)

        # Set mean and variance of Laplace approximation
        mu = psi
        sig = -np.linalg.pinv(S)

        # Update H
        H = 1 - H * np.diag(sig) + nu
        H = H / (mu ** 2 + nu)

    # Prune step
    threshold = 2000
    selector = H < threshold
    X = X[:, selector]
    mu = mu[selector]
    sig = sig[selector, :][:, selector]
    H = H[selector]
    relevant_points = selector.reshape((I, 1))

    I = X.shape[1]
    K = np.zeros((I, I))
    for i in range(I):
        for j in range(I):
            K[i, j] = kernel(X[:, i], X[:, j])

    I_test = X_test.shape[1]
    K_test = np.zeros((I, I_test))
    for i in range(I):
        for j in range(I_test):
            K_test[i, j] = kernel(X[:, i], X_test[:, j])

    mu_a = mu.reshape((1, I)) @ K_test
    var_a_temp = sig @ K_test
    var_a = np.zeros((1, I_test))
    for i in range(I_test):
        var_a[0, i] = K_test[:, i] @ var_a_temp[:, i]

    p_lambda = sigmoid(mu_a / np.sqrt(1 + np.pi / 8 * var_a))
    predictions = p_lambda.reshape(I_test)
    return (predictions, relevant_points)


def _fit_rvc_cost(psi, w, Hd, K):
    I = K.shape[0]
    L = I * (-np.log(fitting.gaussian_pdf(
        psi.reshape((1, psi.size)),
        np.zeros(I),
        np.diag(1 / Hd)
    )[0, 0]))

    predictions = sigmoid(psi @ K)
    for i in range(I):
        y = predictions[i]
        if w[i, 0] == 1:
            L -= np.log(y)
        else:
            L -= np.log(1 - y)
    return L


def _fit_rvc_jac(psi, w, Hd, K):
    I = K.shape[0]
    g = I * Hd * psi
    predictions = sigmoid(psi @ K)
    for i in range(I):
        y = predictions[i]
        g += (y - w[i, 0]) * K[:, i]
    return g


def _fit_rvc_hess(psi, w, Hd, K):
    I = K.shape[1]
    H = I * np.diag(Hd)
    predictions = sigmoid(psi @ K)
    for i in range(I):
        y = predictions[i]
        temp = K[:, i].reshape((I, 1))
        H += y * (1 - y) * (temp @ temp.transpose())
    return H


def fit_incremental_logistic(X, w, X_test, K):
    """Incremental fitting for logistic regression.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            X_test      - test examples for which we need to make predictions.
            K           - the number of incremental stages.
    Output: predictions - 1 * I_test row vector containing the predicted class values for
                          the input data in X_test.
    """
    I = X.shape[1]
    D = X.shape[0] - 1
    phi_0 = 0
    phi = np.zeros(K)
    a = np.zeros(I)
    xi = np.zeros((D + 1, K))

    for k in range(K):
        a = a - phi_0

        initial_x = np.ones(2 + D + 1)
        x = optimize.minimize(
            _fit_inclr_cost,
            initial_x,
            args=(X, w, a),
            method="TNC",
            jac=_fit_inclr_jac,
        ).x

        phi_0 = x[0]
        phi[k] = x[1]
        xi[:, k] = x[2:]

        f = np.arctan(xi[:, k] @ X)
        a += phi_0 + phi[k] * f

    temp = phi.reshape((phi.size, 1)) * np.arctan(xi.transpose() @ X_test)
    act = phi_0 + np.sum(temp, axis=0)
    predictions = sigmoid(act)

    return predictions


def _fit_inclr_cost(x, X, w, a):
    I = X.shape[1]
    f = 0
    temp = x[2:].reshape((1, x.size - 2)) @ X
    for i in range(I):
        y = sigmoid(a[i] + x[0] + x[1] * np.arctan(temp[0, i]))
        if w[i, 0] == 1:
            f -= np.log(y)
        else:
            f -= np.log(1 - y)
    return f


def _fit_inclr_jac(x, X, w, a):
    D = X.shape[0] - 1
    g = np.zeros(2 + D + 1)
    temp1 = x[2:].reshape((1, x.size - 2)) @ X
    temp1 = temp1.reshape(temp1.size)
    y = sigmoid(a + x[0] + x[1] * np.arctan(temp1))
    temp2 = y - w.reshape(w.size)
    g[0] = np.sum(temp2)
    g[1] = np.sum(temp2 * np.arctan(temp1))
    g[2:] = np.sum(temp2 * x[1] * (1 / (1 + temp1 ** 2)) * X, axis=1)
    return g


def fit_logitboost(X, w, X_test, Alpha, K):
    """Logitboost.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            X_test      - test examples for which we need to make predictions.
            Alpha       - (D + 1) * M matrix containing the parameters for the weak 
                          classifiers in its columns.
            K           - the number of incremental stages.
    Output: predictions - 1 * I_test row vector containing the predicted class values for
                          the input data in X_test.
    """
    I = X.shape[1]
    M = Alpha.shape[1]

    a = np.zeros(I)
    phi_0 = 0
    phi = np.zeros(K)
    c = np.zeros(K, dtype=int)

    for k in range(K):
        # Find the best weak classifier
        current_max = -1
        for m in range(M):
            value = 0
            for i in range(I):
                f = Alpha[:, m] @ X[:, i]
                if f < 0:
                    f = 0
                else:
                    f = 1
                value += (sigmoid(a[i]) - w[i, 0]) * f
            value = value ** 2
            if value > current_max:
                current_max = value
                c[k] = m

        a = a - phi_0
        initial_x = np.array([0, 0])
        x = optimize.minimize(
            _fit_logitboost_cost,
            initial_x,
            args=(X, w, a, Alpha[:, c[k]]),
            method="TNC",
            jac=_fit_logitboost_jac,
        ).x
        phi_0 = x[0]
        phi[k] = x[1]

        for i in range(I):
            f = Alpha[:, c[k]] @ X[:, i]
            if f < 0:
                f = 0
            else:
                f = 1
            a[i] += phi_0 + phi[k] * f

    I_test = X_test.shape[1]
    predictions = np.zeros(I_test)
    for i in range(I_test):
        act = phi_0
        for k in range(K):
            f = Alpha[:, c[k]] @ X_test[:, i]
            if f < 0:
                f = 0
            else:
                f = 1
            act += phi[k] * f
        predictions[i] = sigmoid(act)

    return predictions


def _fit_logitboost_cost(x, X, w, a, alpha):
    I = X.shape[1]
    f = 0
    for i in range(I):
        temp = alpha @ X[:, i]
        if (temp < 0):
            temp = 0
        else:
            temp = 1
        y = sigmoid(a[i] + x[0] + x[1] * temp)
        if w[i, 0] == 1:
            f -= np.log(y)
        else:
            f -= np.log(1 - y)
    return f


def _fit_logitboost_jac(x, X, w, a, alpha):
    I = X.shape[1]
    g = np.zeros(2)
    for i in range(I):
        temp = alpha @ X[:, i]
        if (temp < 0):
            temp = 0
        else:
            temp = 1
        y = sigmoid(a[i] + x[0] + x[1] * temp)
        temp2 = y - w[i, 0]
        g[0] += temp2
        g[1] += temp2 * temp
    return g


def fit_multi_logistic(X, w, X_test, class_num):
    """Multi-class logistic regression.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            X_test      - test examples for which we need to make predictions.
            class_num   - the number of classes.
    Output: Predictions - class_num * I_test matrix containing the predicted class values for
                          the data in X_test.
    """
    D = X.shape[0]

    initial_phi = np.ones(D * class_num)
    phi = optimize.minimize(
        _fit_mclr_cost,
        initial_phi,
        args=(X, w, class_num),
        method="Newton-CG",
        jac=_fit_mclr_jac,
        hess=_fit_mclr_hess
    ).x
    Phi = phi.reshape((class_num, D))

    # Predict
    Phi_X_exp = np.exp(Phi @ X_test)
    Predictions = Phi_X_exp / np.sum(Phi_X_exp, axis=0)
    return Predictions


def _fit_mclr_cost(phi, X, w, class_num):
    I = X.shape[1]
    D = X.shape[0]
    L = 0

    Phi = phi.reshape((class_num, D))
    Phi_X_exp = np.exp(Phi @ X)
    Y = Phi_X_exp / np.sum(Phi_X_exp, axis=0)
    for i in range(I):
        L = L - np.log(Y[int(w[i, 0]), i])
    return L


def _fit_mclr_jac(phi, X, w, class_num):
    I = X.shape[1]
    D = X.shape[0]
    g = np.zeros(D * class_num)

    Phi = phi.reshape((class_num, D))
    Phi_X_exp = np.exp(Phi @ X)
    Y = Phi_X_exp / np.sum(Phi_X_exp, axis=0)
    for i in range(I):
        start = 0
        X_i = X[:, i]
        for n in range(class_num):
            temp = (Y[n, i] - (int(w[i, 0]) == n)) * X_i
            g[start:(start + D)] = g[start:(start + D)] + temp
            start += D
    return g


def _fit_mclr_hess(phi, X, w, class_num):
    I = X.shape[1]
    D = X.shape[0]
    H = np.array(
        [[] for i in range(D * class_num)]
    ).reshape((0, D * class_num))
    HH = [
        [np.zeros((D, D)) for n in range(class_num)]
        for m in range(class_num)
    ]

    Phi = phi.reshape((class_num, D))
    Phi_X_exp = np.exp(Phi @ X)
    Y = Phi_X_exp / np.sum(Phi_X_exp, axis=0)
    for i in range(I):
        X_i = X[:, i].reshape((D, 1))
        for n in range(class_num):
            for m in range(class_num):
                temp = Y[m, i] * ((m == n) - Y[n, i]) * (X_i @ X_i.transpose())
                HH[m][n] = HH[m][n] + temp

    # Assemble final Hessian
    for n in range(class_num):
        H_n = np.array([[] for i in range(D)])
        for m in range(class_num):
            H_n = np.append(H_n, HH[n][m], axis=1)
        H = np.append(H, H_n, axis=0)
    return H


def fit_multi_logistic_tree(X, w, X_test, J, G, K):
    """Multi-class logistic classification tree.

    Input:  X           - (D + 1) * I training data matrix, where D is the dimensionality
                          and I is the number of training examples.
            w           - I * 1 vector containing world states for each example.
            X_test      - test examples for which we need to make predictions.
            J           - the number of nodes in the tree.
            G           - linear functions in columns used as classifiers.
            K           - the number of classes.
    Output: Predictions - K * I_test matrix containing the predicted class values for
                          the data in X_test.
    """
    I = X.shape[1]
    I_test = X_test.shape[1]
    M = G.shape[1]
    c = np.zeros(J, dtype=int)
    Lambda = np.zeros((K, J + 1))
    G_T = G.transpose()
    GX = sigmoid(G_T @ X) > 0.5
    GX_test = sigmoid(G_T @ X_test) > 0.5
    Predictions = np.zeros((K, I_test))

    queue = deque([np.arange(I)])

    # For each node in the tree
    for j in range(J):
        current_data = queue.popleft()
        II = current_data.size
        l = np.zeros(M)
        for m in range(M):
            # Count frequency for k-th class in left and right branch
            n_l = np.zeros(K)
            n_r = np.zeros(K)
            for i in range(II):
                ii = current_data[i]
                k = int(w[ii, 0])
                if GX[m, ii] == True:
                    n_r[k] += 1
                else:
                    n_l[k] += 1

            # Compute the log likelihood
            sum_l = np.sum(n_l)
            sum_r = np.sum(n_r)
            # If this classifier cannot separate the data, then it
            # should not be applied
            if sum_l == 0 or sum_r == 0:
                l[m] = float("inf")
            else:
                # Protect against zero values
                n_l += 1
                n_r += 1
                sum_l += K
                sum_r += K

                # Compute log likelihood
                norm_l = np.log(n_l / sum_l)
                norm_r = np.log(n_r / sum_r)
                l[m] = np.sum(norm_l) + np.sum(norm_r)
        # Store index of best clasifier for this node
        c[j] = np.argmin(l)
        # Partition in two sets
        right = GX[c[j], current_data]
        queue.append(current_data[right])
        queue.append(current_data[np.logical_not(right)])

    # Recover categorical paramters at J+1 leaves
    for p in range(J + 1):
        current_data = queue.popleft()
        II = current_data.size
        n = np.zeros(K)
        for i in range(II):
            ii = current_data[i]
            k = int(w[ii])
            n[k] += 1
        Lambda[:, p] = n / np.sum(n)
    
    # Predict by moving each test example from the root to the
    # bottom of the tree, and then taking the corresponding
    # categorical distribution from the leaf node
    for i in range(I_test):
        j = 1
        right = 0
        tree_level = 0
        while True:
            right = GX_test[c[j - 1], i]
            new_j = 2 * j
            if right == True:
                new_j = new_j + 1
            if new_j > J:
                break
            else:
                j = new_j
                tree_level += 1
        node_index_in_last_level_left_to_right = j - 2 ** tree_level
        if right == False:
            leaf_node = node_index_in_last_level_left_to_right * 2 + 1
        else:
            leaf_node = node_index_in_last_level_left_to_right * 2 + 2
        Predictions[:, i] = Lambda[:, leaf_node - 1]
    return Predictions