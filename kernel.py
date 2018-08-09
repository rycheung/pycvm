import numpy as np


def gaussian(x_i, x_j, p_lambda):
    """Gaussian kernel function.

    Input:  x_i      - a 1D array.
            x_j      - a 1D array.
            p_lambda - kernel parameter.
    Output: f        - result.
    """
    x_diff = x_i - x_j
    temp = x_diff @ x_diff
    f = np.exp(-0.5 * temp / (p_lambda ** 2))
    return f


def linear(x_i, x_j):
    """Linear kernel function.

    Input:  x_i      - a 1D array.
            x_j      - a 1D array.
    Output: f        - result.
    """
    return x_i @ x_j
