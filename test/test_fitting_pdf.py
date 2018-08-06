import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import fitting
import numpy as np
import matplotlib.pyplot as plt

argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_fitting_pdf.py t_pdf|gamma_pdf")
    sys.exit(-1)

if argv[1] == "t_pdf":
    plt.figure("T-distribution")

    x = np.arange(-5, 5, 0.01)
    mu = 0
    sig = 1
    px1 = fitting.t_pdf(x, mu, sig, 1)
    px2 = fitting.t_pdf(x, mu, sig, 2)
    px3 = fitting.t_pdf(x, mu, sig, 5)
    norm_pdf = fitting.gaussian_pdf(x.reshape((x.size, 1)),
                                    np.array([mu]),
                                    np.array([[sig]]))[:, 0]

    plt.subplot(121)
    line1, = plt.plot(x, px1, 'b')
    line2, = plt.plot(x, px2, 'g')
    line3, = plt.plot(x, px3, 'y')
    norm_line, = plt.plot(x, norm_pdf, 'r')
    plt.axis([-5, 5, 0, 0.6])
    plt.legend([line1, line2, line3, norm_line],
               ['t-distribution, nu = 1', 't-distribution, nu = 2',
                't-distribution, nu = 5', 'normal distribution'])

    plt.subplot(122)
    line1, = plt.semilogy(x, px1, 'b')
    line2, = plt.semilogy(x, px2, 'g')
    line3, = plt.semilogy(x, px3, 'y')
    norm_line, = plt.semilogy(x, norm_pdf, 'r')
    plt.axis([-5, 5, 0, 100])
    plt.legend([line1, line2, line3, norm_line],
               ['t-distribution, nu = 1', 't-distribution, nu = 2',
                't-distribution, nu = 5', 'normal distribution'])

    plt.show()

elif argv[1] == "gamma_pdf":
    plt.figure("Gamma distribution")

    x = np.arange(0, 10, 0.01)
    px1 = fitting.gamma_pdf(x, 5, 4)
    px2 = fitting.gamma_pdf(x, 10, 4)
    px3 = fitting.gamma_pdf(x, 20, 4)
    px4 = fitting.gamma_pdf(x, 30, 4)

    plt.subplot(121)
    line1, = plt.plot(x, px1, 'b')
    line2, = plt.plot(x, px2, 'g')
    line3, = plt.plot(x, px3, 'y')
    line4, = plt.plot(x, px4, 'r')
    plt.axis([0, 10, 0, 1])
    plt.legend([line1, line2, line3, line4],
               ['alpha=5,  beta=4', 'alpha=10, beta=4',
                'alpha=20, beta=4', 'alpha=30, beta=4'])

    x = np.arange(0, 30, 0.01)
    px1 = fitting.gamma_pdf(x, 20, 8)
    px2 = fitting.gamma_pdf(x, 20, 4)
    px3 = fitting.gamma_pdf(x, 20, 2)
    px4 = fitting.gamma_pdf(x, 20, 1)

    plt.subplot(122)
    line1, = plt.plot(x, px1, 'b')
    line2, = plt.plot(x, px2, 'g')
    line3, = plt.plot(x, px3, 'y')
    line4, = plt.plot(x, px4, 'r')
    plt.axis([0, 30, 0, 1])
    plt.legend([line1, line2, line3, line4],
               ['alpha=20, beta=8', 'alpha=20, beta=4',
                'alpha=20, beta=2', 'alpha=20, beta=1'])

    plt.show()

elif argv[1] == "mul_t_pdf":
    mu = np.array([1, 3])
    sig = np.array([[2, 1.3], [2.4, 7]])
    nu = 5

    XX, YY = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))
    x = XX.reshape((XX.size, 1))
    y = YY.reshape((XX.size, 1))
    xy_matrix = np.append(x, y, axis=1)

    results = fitting.mul_t_pdf(xy_matrix, mu, sig, nu)
    results = results.reshape((XX.shape[0], XX.shape[1]))

    plt.figure("Multivariate t-distribution")
    plt.axis([-10, 10, -10, 10])
    plt.contour(XX, YY, results)
    plt.show()
