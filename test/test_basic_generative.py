import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.abspath(__file__), '..', '..')
    )
)


import classification
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sig):
    return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-0.5 * (x - mu) ** 2 / (sig ** 2))


N = 1000
r1 = np.random.normal(-2, 0.5, (N, 1))
r2 = np.random.normal(2, 0.5, (N, 1))
r = np.append(r1, r2, axis=0)
y = np.zeros((N, 1))

plt.figure("Basic generative classifier")
plt.subplot(221)
plt.plot(r1, y, 'bo')
plt.plot(r2, y, 'ro')
plt.axis([-4, 4, -0.1, 0.1])

class_column = np.append(np.zeros((N, 1)), np.ones((N, 1)), axis=0)
x_train = np.c_[r, class_column]
x_test = np.transpose([np.arange(-4, 4, 0.01)])
(p_lambda, mu, sig, posterior) = classification.basic_generative(x_train, x_test, 2)

plt.subplot(222)
plt.plot(x_test, posterior[:, 0], 'b')
plt.plot(x_test, posterior[:, 1], 'r')
plt.axis([-4, 4, 0, 1.5])

plt.subplot(223)
plt.bar(np.arange(2), p_lambda[:, 0], 0.8, align="center", color="blue")
plt.axis([-0.5, 1.5, 0, 1])

x = np.arange(-4, 4, 0.01)
plt.subplot(224)
norm_pdf1 = gaussian(x, mu[0, 0], sig[0][0, 0])
norm_pdf2 = gaussian(x, mu[1, 0], sig[1][0, 0])
plt.plot(x, norm_pdf1, 'b')
plt.plot(x, norm_pdf2, 'r')
plt.axis([-4, 4, 0, 2])

plt.show()
