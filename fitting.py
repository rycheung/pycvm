import numpy as np
from scipy.special import gamma as gamma_func

def mle_norm(x):
    """Maximum likelihood learning for normal distribution

    Input:  Training data (x)
    Output: Maximum likelihood estimates of parameters (mu, var)
    """
    I = x.size
    mu = np.sum(x) / I
    var = np.sum((x - mu) ** 2) / I
    return (mu, var)

def map_norm(x, alpha, beta, gamma, delta):
    """MAP learning for normal distribution

    Input:  Training data (x),
            Hyperparameters alpha, beta, gamma, delta
    Output: MAP estimates of params (mu, var)
    """
    I = x.size
    mu = (np.sum(x) + gamma * delta) / (I + gamma)
    var_numerator = np.sum((x - mu) ** 2) + 2 * beta + gamma * (delta - mu) ** 2
    var_denominator = I + 3 + 2 * alpha
    var = var_numerator / var_denominator
    return (mu, var)

def by_norm(x, alpha_prior, beta_prior, gamma_prior, delta_prior, x_test):
    """Bayesian approach to normal distribution

    Input:  Training data (x),
            Hyperparameters (alpha_prior, beta_prior, gamma_prior, delta_prior),
            Test data (x_test)
    Output: Posterior parameters (alpha_post, beta_post, gamma_post, delta_post),
            Predictive distribution (x_prediction)
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
    """Maximum likelihood learning for categorical distribution

    Input:  Training data (x),
            The number of categorical parameters (K)
    Output: ML estimate of categorical parameters (theta)
    """
    counts = np.histogram(x, np.arange(1, K + 2))[0]
    theta = counts / x.size
    return theta

def map_cat(x, alpha):
    """MAP learning for categorical distribution with conjugate prior

    Input:  Training data (x),
            Hyperparameters (alpha)
    Output: MAP estimate of categorical parameters (theta)
    """
    K = alpha.size
    counts = np.histogram(x, np.arange(1, K + 2))[0]
    tmp = counts + alpha - 1
    theta = tmp / np.sum(tmp)
    return theta

def by_cat(x, alpha_prior):
    """MAP learning for categorical distribution with conjugate prior

    Input:  Training data (x),
            Hyperparameters (alpha_prior)
    Output: Posterior parameters (alpha_post),
            Predictive distribution (x_prediction)
    """
    K = alpha_prior.size
    counts = np.histogram(x, np.arange(1, K + 2))[0]
    alpha_post = counts + alpha_prior
    prediction = alpha_post / np.sum(alpha_post)
    return (alpha_post, prediction)