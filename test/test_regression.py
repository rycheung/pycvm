import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.abspath(__file__), '..', '..')
    )
)

import regression
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sig):
    return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-0.5 * (x - mu) ** 2 / (sig ** 2))


argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_regression.py fit_linear|")
    sys.exit(-1)

if argv[1] == 'fit_linear':
    I = 50
    x = np.linspace(1, 9, I)
    phi = np.array([[7], [-0.5]])
    sig = 0.6
    w = np.zeros((I, 1))
    X = np.append(np.ones((I, 1)), x.reshape((I, 1)), axis=1)
    X_phi = X @ phi
    r = sig * np.random.standard_normal((I, 1))
    for i in range(I):
        w[i] = X_phi[i] + r[i]

    fit_phi, fit_sig = regression.fit_linear(X.transpose(), w)
    granularity = 500
    domain = np.linspace(0, 10, granularity)
    X, Y = np.meshgrid(domain, domain)
    XX = np.append(np.ones((granularity, 1)),
                   domain.reshape((granularity, 1)), axis=1)
    temp = XX @ fit_phi
    Z = np.zeros((granularity, granularity))
    for j in range(granularity):
        mu = temp[j, 0]
        for i in range(granularity):
            ww = domain[i]
            Z[i, j] = gaussian(ww, mu, fit_sig)

    plt.figure("Fit linear regression")
    res = plt.pcolor(X, Y, Z)
    plt.scatter(x, w, edgecolors="w")
    plt.show()


else:
    print("Usage: python test_regression.py fit_linear|")
