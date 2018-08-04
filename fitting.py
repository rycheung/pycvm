import numpy as np
from scipy.special import gamma as gamma_func

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
    var_numerator = np.sum((x - mu) ** 2) + 2 * beta + gamma * (delta - mu) ** 2
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
    gamma_int  = gamma_post + 1

    print(alpha_post, alpha_int)

    # Predict values for x_test
    x_prediction_num = np.sqrt(gamma_post) * np.float_power(beta_post, alpha_post) * gamma_func(alpha_int)
    x_prediction_den = np.sqrt(2 * np.pi * gamma_int) * np.float_power(beta_int, alpha_int) * gamma_func(alpha_post)
    x_prediction = x_prediction_num / x_prediction_den

    return (alpha_post, beta_post, gamma_post, delta_post, x_prediction)

def mle_cat(x, K):
    """Maximum likelihood learning for categorical distribution.

    Input:  x     - training data.
            K     - the number of categorical parameters.
    Output: theta - ML estimate of categorical parameters.
    """
    counts = np.histogram(x, np.arange(1, K + 2))[0]
    theta = counts / x.size
    return theta

def map_cat(x, alpha):
    """MAP learning for categorical distribution with conjugate prior.

    Input:  x     - training data.
            alpha - hyperparameters of Dirichlet distribution.
    Output: theta - MAP estimate of categorical parameters.
    """
    K = alpha.size
    counts = np.histogram(x, np.arange(1, K + 2))[0]
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
    counts = np.histogram(x, np.arange(1, K + 2))[0]
    alpha_post = counts + alpha_prior
    prediction = alpha_post / np.sum(alpha_post)
    return (alpha_post, prediction)