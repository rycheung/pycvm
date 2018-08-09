import numpy as np


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
