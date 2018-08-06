import numpy as np
from scipy.special import gamma as gamma_func


def gaussian(X, mu, sig):
    k = 1 / ((2 * np.pi) ** (mu.size / 2) * np.sqrt(np.linalg.det(sig)))
    sig_inv = np.linalg.pinv(sig)
    exp_factors = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        exp_factors[i] = np.exp(-0.5 * np.dot(np.dot((X[i] - mu),
                                                     sig_inv), np.transpose([X[i] - mu])))
    return k * exp_factors


def mle_norm(x):
    """Maximum likelihood learning for normal distribution.

    Input:  x   - training data.
    Output: mu  - mean of the normal distribution.
            var - variance of the normal distribution.
    """
    I = x.size
    mu = np.sum(x) / I
    var = np.sum((x - mu) ** 2) / I
    return (mu, var)


def map_norm(x, alpha, beta, gamma, delta):
    """MAP learning for normal distribution.

    Input:  x     - training data.
            alpha - hyperparameter of normal-scaled inverse gamma distribution.
            beta  - hyperparameter of normal-scaled inverse gamma distribution.
            gamma - hyperparameter of normal-scaled inverse gamma distribution.
            delta - hyperparameter of normal-scaled inverse gamma distribution.
    Output: mu    - mean of the normal distribution.
            var   - variance of the normal distribution.
    """
    I = x.size
    mu = (np.sum(x) + gamma * delta) / (I + gamma)
    var_numerator = np.sum((x - mu) ** 2) + 2 * beta + \
        gamma * (delta - mu) ** 2
    var_denominator = I + 3 + 2 * alpha
    var = var_numerator / var_denominator
    return (mu, var)


def by_norm(x, alpha_prior, beta_prior, gamma_prior, delta_prior, x_test):
    """Bayesian approach to normal distribution.

    Input:  x            - training data.
            alpha_prior  - hyperparameter of normal-scaled inverse gamma distribution.
            beta_prior   - hyperparameter of normal-scaled inverse gamma distribution.
            gamma_prior  - hyperparameter of normal-scaled inverse gamma distribution.
            delta_prior  - hyperparameter of normal-scaled inverse gamma distribution.
            x_test       - test data.
    Output: alpha_post   - posterior parameters.
            beta_post    - posterior parameters.
            gamma_post   - posterior parameters.
            delta_post   - posterior parameters.
            x_prediction - predictive distribution.
    """
    I = x.size
    x_sum = np.sum(x)

    # Compute normal inverse gamma posterior over normal parameters
    alpha_post = alpha_prior + I / 2
    beta_post = np.sum(x ** 2) / 2 + beta_prior + gamma_prior * (delta_prior ** 2) / 2 - \
        (gamma_prior * delta_prior + x_sum) ** 2 / (2 * (gamma_prior + I))
    gamma_post = gamma_prior + I
    delta_post = (gamma_prior * delta_prior + x_sum) / (gamma_prior + I)

    # Compute intermediate parameters
    alpha_int = alpha_post + 1 / 2
    beta_int = x_test ** 2 / 2 + beta_post + gamma_post * delta_post ** 2 / 2 - \
        (gamma_post * delta_post + x_test) ** 2 / (2 * (gamma_post + 1))
    gamma_int = gamma_post + 1

    # Predict values for x_test
    x_prediction_num = np.sqrt(
        gamma_post) * np.float_power(beta_post, alpha_post) * gamma_func(alpha_int)
    x_prediction_den = np.sqrt(2 * np.pi * gamma_int) * \
        np.float_power(beta_int, alpha_int) * gamma_func(alpha_post)
    x_prediction = x_prediction_num / x_prediction_den

    return (alpha_post, beta_post, gamma_post, delta_post, x_prediction)


def mle_cat(x, K):
    """Maximum likelihood learning for categorical distribution.

    Input:  x     - training data.
            K     - the number of categorical parameters.
    Output: theta - ML estimate of categorical parameters.
    """
    counts = np.histogram(x, np.arange(K + 1))[0]
    theta = counts / x.size
    return theta


def map_cat(x, alpha):
    """MAP learning for categorical distribution with conjugate prior.

    Input:  x     - training data.
            alpha - hyperparameters of Dirichlet distribution.
    Output: theta - MAP estimate of categorical parameters.
    """
    K = alpha.size
    counts = np.histogram(x, np.arange(K + 1))[0]
    tmp = counts + alpha - 1
    theta = tmp / np.sum(tmp)
    return theta


def by_cat(x, alpha_prior):
    """Bayesian approach for categorical distribution.

    Input:  x            - training data.
            alpha_prior  - hyperparameters of Dirichlet distribution.
    Output: alpha_post   - Posterior parameters.
            x_prediction - predictive distribution.
    """
    K = alpha_prior.size
    counts = np.histogram(x, np.arange(K + 1))[0]
    alpha_post = counts + alpha_prior
    prediction = alpha_post / np.sum(alpha_post)
    return (alpha_post, prediction)


def em_mog(x, K, precision):
    """Fitting mixture of Gaussians using EA algorithm.

    Input:  x         - training data.
            K         - the number of Gaussians in the mixture.
            precision - the algorithm stops when the difference between the 
                        previous and the new likelihood is smaller than precision.
    Output: p_lambda  - p_lambda[k] is the weight for the k-th Gaussian.
            mu        - mu[k, :] is the mean for the k-th Gaussian.
            sig       - sig[k] is the covariance matrix for the k-th Gaussian.
    """
    I = x.shape[0]
    dimensionality = x.shape[1]

    # Init p_lamda
    p_lambda = np.array([1 / K] * K)

    # Init mu with K random samples
    K_choice = np.random.choice(I, K, replace=False)
    mu = x[np.ix_(K_choice, np.arange(dimensionality))]

    # Init sig with the variance of the dataset
    dataset_mean = np.sum(x, axis=0) / I
    dataset_variance = np.zeros((dimensionality, dimensionality))
    for i in range(I):
        data_sample = x[i, :].reshape((1, dimensionality))
        mat = data_sample - dataset_mean
        mat = mat.transpose() @ mat
        dataset_variance += mat
    dataset_variance /= I
    sig = [dataset_variance.copy() for x in range(K)]

    # Expectation maximization algorithm
    iterations = 0
    previous_L = 2000000
    L = 1000000
    while np.absolute(L - previous_L) >= precision:
        previous_L = L
        iterations += 1

        # Expectation step
        l = np.zeros((I, K))
        r = np.zeros((I, K))
        for k in range(K):
            l[:, k] = p_lambda[k] * gaussian(x, mu[k, :], sig[k])[:, 0]

        s = np.sum(l, axis=1).reshape((I, 1))
        for i in range(I):
            r[i, :] = l[i, :] / s[i, 0]

        # Maximization step
        r_summed_rows = np.sum(r, axis=0)
        r_summed_all = np.sum(r_summed_rows)

        # Update p_lambda
        p_lambda = r_summed_rows / r_summed_all

        for k in range(K):
            # Update mu
            mu[k, :] = np.sum(r[:, k].reshape((I, 1)) * x,
                              axis=0) / r_summed_rows[k]

            # Update sig
            delta = (x - mu[k, :])
            numerator = np.zeros((dimensionality, dimensionality))
            for i in range(I):
                vec = delta[i].reshape((1, dimensionality))
                numerator += r[i, k] * vec.transpose() @ vec
            sig[k] = numerator / r_summed_rows[k]

        # Compute the log likelihood L
        mat = np.zeros((I, K))
        for k in range(K):
            mat[:, k] = p_lambda[k] * gaussian(x, mu[k, :], sig[k])[:, 0]

        L = np.sum(np.sum(mat, axis=1))

    return (p_lambda, mu, sig)
