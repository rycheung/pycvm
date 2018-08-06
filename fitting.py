import numpy as np
from scipy.special import \
    gamma as gamma_func, \
    gammaln as gammaln_func, \
    digamma as digamma_func \



def gaussian_pdf(X, mu, sig):
    """ Multivariate Gaussian distribution.
    Input:  X   - input data.
            mu  - mean of the distribution.
            sig - covariance matrix of the distribution.
    Output: px  - output data.
    """
    k = 1 / ((2 * np.pi) ** (mu.size / 2) * np.sqrt(np.linalg.det(sig)))
    sig_inv = np.linalg.pinv(sig)
    exp_factors = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        exp_factors[i] = np.exp(-0.5 * (X[i] - mu) @
                                sig_inv @ np.transpose([X[i] - mu]))
    px = k * exp_factors
    return px


def t_pdf(x, mu, sig, nu):
    """ Univariate t-distribution.
    Input:  X   - input data.
            mu  - mean of the distribution.
            sig - sacle of the distribution.
            nu  - degrees of freedom.
    Output: px  - output data.
    """
    px = gamma_func((nu + 1) / 2) / \
        (np.sqrt(nu * np.pi * sig) * gamma_func(nu / 2))
    px = px * np.float_power(1 + (x - mu) ** 2 / (nu * sig), (-nu - 1) / 2)
    return px


def gamma_pdf(x, alpha, beta):
    """ Univariate gamma-distribution.
    Input:  X   - input data.
            alpha  - parameter of the distribution.
            beta - parameter of the distribution.
    Output: px  - output data.
    """
    px = np.float_power(beta, alpha) / gamma_func(alpha)
    px = px * np.exp(-beta * x) * np.float_power(x, alpha - 1)
    return px


def mul_t_pdf(x, mu, sig, nu):
    """ Multivariate t-distribution.
    Input:  X   - input data.
            mu  - mean of the distribution.
            sig - scale matrix of the distribution.
            nu  - degrees of freedom.
    Output: px  - output data.
    """
    mu = mu.reshape((mu.size, 1))
    D = mu.size
    # `gammaln` is used instead of gamma to avoid overflow.
    c = np.exp(gammaln_func((nu + D) / 2) - gammaln_func(nu / 2))
    c = c / (np.float_power(nu * np.pi, D / 2) * np.sqrt(np.linalg.det(sig)))

    I = x.shape[0]
    delta = np.zeros((I, 1))
    x_minus_mu = x - mu.transpose()
    temp = x_minus_mu @ np.linalg.pinv(sig)
    for i in range(I):
        delta[i, 0] = temp[i, :].reshape((1, D)) @ \
            x_minus_mu[i].reshape((D, 1))

    px = np.float_power(1 + delta / nu, -(nu + D) / 2)
    px = px * c

    return px


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
    """Fitting mixture of Gaussians using EM algorithm.

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
            l[:, k] = p_lambda[k] * gaussian_pdf(x, mu[k, :], sig[k])[:, 0]

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
            mat[:, k] = p_lambda[k] * gaussian_pdf(x, mu[k, :], sig[k])[:, 0]

        L = np.sum(np.sum(mat, axis=1))

    return (p_lambda, mu, sig)


def em_t_distribution(x, precision):
    """Fitting mixture of Gaussians using EM algorithm.

    Input:  x         - training data.
            precision - the algorithm stops when the difference between the
                        previous and the new likelihood is smaller than precision.
    Output: mu        - mean of the distribution.
            sig       - scale matrix of the distribution.
            nu        - degrees of freedom
    """
    I = x.shape[0]
    D = x.shape[1]

    # Init mu to the mean of the dataset
    mu = np.sum(x, axis=0) / I
    mu = mu.reshape((1, D))

    # Init sig to the covariance of the dataset
    sig = np.zeros(D, D)
    x_minus_mu = x - mu
    for i in range(I):
        mat = x_minus_mu[i, :].reshape((D, 1))
        mat = mat * mat.transpose()
        sig += mat
    sig /= I

    # Init nu to 1000
    nu = 1000

    iterations = 0
    previous_L = 2000000
    L = 1000000
    delta = np.zeros((I, 1))
    while np.absolute(L - previous_L) >= precision:
        previous_L = L
        iterations += 1

        # Expectation step
        # Compute delta
        x_minus_mu = x - mu
        temp = x_minus_mu @ np.linalg.pinv(sig)
        for i in range(I):
            delta[i, 0] = temp[i, :].reshape((1, D)) @ \
                x_minus_mu[i].reshape((D, 1))

        # Compute E_hi and E_log_hi
        nu_plus_delta = nu + delta
        E_hi = (nu + D) / nu_plus_delta
        E_log_hi = digamma_func((nu + D) / 2) - np.log(nu_plus_delta / 2)

        # Maximization step
        # Update mu
        E_hi_sum = np.sum(E_hi)
        mu = np.sum(x * delta, axis=0) / E_hi_sum

        # Update sig
        x_minus_mu = x - mu
        sig = np.zeros(D, D)
        for i in range(I):
            xmm = x_minus_mu[i, :].reshape((D, 1))
            sig += E_hi[i] * xmm * xmm.transpose()
        sig /= E_hi_sum

        # Update nu by minimizing a cost function with line search
        nu = _fit_t_minimize_cost(E_hi, E_log_hi)

        # Compute data log likelihood
        temp = x_minus_mu @ np.linalg.pinv(sig)
        for i in range(I):
            delta[i, 0] = temp[i, :].reshape((1, D)) @ \
                x_minus_mu[i].reshape((D, 1))
        L = I * gammaln_func((nu + D) / 2) - \
            I * D * np.log(nu * np.pi) / 2 - \
            I * np.log(np.linalg.det(sig)) / 2 - \
            I * gammaln_func(nu / 2)
        L = L - (nu + D) * np.sum(np.log(1 + delta / nu)) / 2


def _fit_t_minimize_cost(E_hi, E_log_hi):
    return 1
