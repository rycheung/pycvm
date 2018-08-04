import numpy as np

def gaussian(X, mu, sig):
    k = 1 / ((2 * np.pi) ** (mu.size / 2) * np.sqrt(np.linalg.det(sig)))
    sig_inv = np.linalg.pinv(sig)
    exp_factors = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        exp_factors[i] = np.exp(-0.5 * np.dot(np.dot((X[i] - mu), sig_inv), np.transpose([X[i] - mu])))
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
