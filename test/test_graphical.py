import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.abspath(__file__), '..', '..')
    )
)

import fitting
import graphical
import numpy as np
import matplotlib.pyplot as plt

argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_graphical.py gibbs_sampling")
    sys.exit(-1)

if argv[1] == 'gibbs_sampling':
    Phi = [
        lambda x:fitting.gaussian_pdf(
            x.reshape((1, x.size)), 
            np.array([0, 0]), 
            np.array([[2, 0], [0, 2]])
        )[0, 0]
    ]
    S = [np.array([0, 1])]
    D = 2
    Values = np.arange(-5, 5, 0.5)
    T = 5000
    Samples = graphical.gibbs_sampling(Phi, S, D, Values, T)
    discard = 500  # discard first samples
    numberOfSamples = (T - discard) / 5  # take every 5th sample
    Samples = Samples[:, discard:]
    temp = np.arange(0, Samples.shape[1], 5)
    Samples = Samples[:, temp]

    # Plot 2D histogram
    K = Values.size
    H = np.zeros((K, K))
    I = Samples.shape[1]
    Samples[1, :] = K - 1 - Samples[1, :]
    for i in range(I):
        col = Samples[0, i]
        row = Samples[1, i]
        H[row, col] += 1
    plt.imshow(H)
    plt.show()

else:
    print("Usage: python test_graphical.py gibbs_sampling")

