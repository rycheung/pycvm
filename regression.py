import numpy as np
from scipy import optimize
import fitting


def fit_linear(X, w):
    """ML fitting of linear regression model.

    Input:  X   - (D + 1) * I training data matrix, where D is the dimensionality
                  and I is the number of training examples.
            w   - I * 1 vector containing world states for each example.
    Output: phi - (D + 1) * 1 vector containing the linear function coefficients.
            sig - variance.
    """
    phi = np.linalg.pinv(X @ X.transpose()) @ X @ w
    temp = w - (X.transpose() @ phi)
    sig = temp.transpose() @ temp / X.shape[1]
    return (phi, sig)


def fit_by_linear(X, w, var_prior, X_test):
    """Bayesian linear regression.

    Input:  X         - (D + 1) * I training data matrix, where D is the dimensionality
                        and I is the number of training examples.
            w         - I * 1 vector containing world states for each example.
            var_prior - scale factor for the prior spherical covariance.
            X_test    - test examples for which we need to make predictions
    Output: mu_test   - I_test * 1 vector containing the means of the distribution 
                        for the test examples.
            var_test  - I_test * 1 vector containing the variance of the distribution 
                        for the test examples.
            var       - can later be used for plotting the posterior over phi.
            A_inv     - can later be used for plotting the posterior over phi.
    """
    D = X.shape[0] - 1
    I = X.shape[1]
    I_test = X_test.shape[1]

    # Compute the variance, using the range [0, variance of world values]
    # Constrain var to be positive, by expressing it as var=sqrt(var)^2
    mu_world = np.sum(w) / I
    var_world = np.sum((w - mu_world) ** 2) / I
    var = optimize.fminbound(
        _fit_blr_cost, 0, var_world,
        (X, w.reshape((1, I)), var_prior)
    )

    # Compute A_inv
    A_inv = 0
    if D < I:
        A_inv = np.linalg.pinv(
            X @ X.transpose() / var + np.eye(D + 1) / var_prior
        )
    else:
        A_inv = np.eye(D + 1) + \
            X @ \
            np.linalg.pinv(X.transpose() @ X + var / var_prior * np.eye(I)) @ \
            X.transpose()
        A_inv = var_prior * A_inv

    temp = X_test.transpose() @ A_inv
    # Compute mu_test
    mu_test = temp @ X @ w / var

    # Compute var_test
    var_test = np.zeros((I_test, 1))
    for i in range(I_test):
        var_test[i, 0] = temp[i, :].reshape((1, D + 1)) @ \
            X_test[:, i].reshape((D + 1, 1)) + var

    return (mu_test, var_test, var, A_inv)


def _fit_blr_cost(var, X, w, var_prior):
    I = X.shape[1]
    covariance = var_prior * (X.transpose() @ X) + \
        np.sqrt(var) ** 2 * np.eye(I)
    f = fitting.gaussian_pdf(w, np.zeros(I), covariance)[0, 0]
    f = -np.log(f)
    return f
