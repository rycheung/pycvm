import sys
import numpy as np
import matplotlib.pyplot as plt
import fitting

argv = sys.argv
if len(argv) != 2:
    print("Usage: python test_fitting.py mle_norm|map_norm|by_norm|mle_cat|map_cat|by_cat")
    sys.exit(-1)

def gaussian(x, mu, sig):
    return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-0.5 * (x - mu) ** 2 / (sig ** 2))

if argv[1] == "mle_norm":
    I_list = [5, 30, 1000, 1000000]
    for index, I in enumerate(I_list):
        original_mu = 5
        original_sig = 8
        r = np.random.normal(original_mu, original_sig, I)

        estimated_mu, estimated_var = fitting.mle_norm(r)
        estimated_sig = np.sqrt(estimated_var)

        mu_error = np.abs(original_mu - estimated_mu)
        sig_error = np.abs(original_sig - estimated_sig)
        print("Mu error:", mu_error, end="")
        print("\nSig error:", sig_error, end="")

        x = np.arange(-20, 30, 0.01)
        original = gaussian(x, original_mu, original_sig)
        estimated = gaussian(x, estimated_mu, estimated_sig)

        plt.figure("ML Norm")
        plt.subplot(221 + index)
        plt.plot(x, original, 'b')
        plt.plot(x, estimated, 'r')
    plt.show()

elif argv[1] == "map_norm":
    I_list = [5, 30, 1000, 1000000]
    for index, I in enumerate(I_list):
        original_mu = 5
        original_sig = 8
        r = np.random.normal(original_mu, original_sig, I)

        estimated_mu, estimated_var = fitting.map_norm(r, 1, 1, 1, 0)
        estimated_sig = np.sqrt(estimated_var)

        mu_error = np.abs(original_mu - estimated_mu)
        sig_error = np.abs(original_sig - estimated_sig)
        print("Mu error:", mu_error, end="")
        print("\nSig error:", sig_error, end="")

        x = np.arange(-20, 30, 0.01)
        original = gaussian(x, original_mu, original_sig)
        estimated = gaussian(x, estimated_mu, estimated_sig)

        plt.figure("MAP Norm")
        plt.subplot(221 + index)
        plt.plot(x, original, 'b')
        plt.plot(x, estimated, 'r')
    plt.show()

elif argv[1] == 'by_norm':
    I_list = [5, 30, 60, 110]
    for index, I in enumerate(I_list):
        original_mu = 5
        original_sig = 8
        r = np.random.normal(original_mu, original_sig, I)

        x_test = np.arange(-20, 30, 0.01)
        alpha_post, beta_post, gamma_post, delta_post, x_prediction = fitting.by_norm(r, 1, 1, 1, 0, x_test)
        
        original = gaussian(x_test, original_mu, original_sig)
        estimated = x_prediction

        plt.figure("Bayesian Norm")
        plt.subplot(221 + index)
        plt.plot(x_test, original, 'b')
        plt.plot(x_test, estimated, 'r')
    plt.show()

elif argv[1] == 'mle_cat':
    I_list = [5, 30, 500, 1000000]
    for index, I in enumerate(I_list):
        original_probabilities = np.array([0.25, 0.15, 0.1, 0.1, 0.15, 0.25])
        r = np.random.choice(np.arange(1, 7), I, p=original_probabilities)
        estimated_probabilities = fitting.mle_cat(r, 6)
        
        plt.figure("ML Cat")
        plt.subplot(421 + index * 2)
        plt.bar(np.arange(1, 7), original_probabilities, 1, align="center", color="blue")
        plt.subplot(422 + index * 2)
        plt.bar(np.arange(1, 7), estimated_probabilities, 1, align="center", color="r")
    plt.show()

elif argv[1] == 'map_cat':
    I_list = [5, 30, 500, 1000000]
    for index, I in enumerate(I_list):
        original_probabilities = np.array([0.25, 0.15, 0.1, 0.1, 0.15, 0.25])
        r = np.random.choice(np.arange(1, 7), I, p=original_probabilities)

        prior = np.array([10, 5, 4, 4, 5, 10])
        estimated_probabilities = fitting.map_cat(r, prior)
        
        plt.figure("MAP Cat")
        plt.subplot(421 + index * 2)
        plt.bar(np.arange(1, 7), original_probabilities, 1, align="center", color="blue")
        plt.subplot(422 + index * 2)
        plt.bar(np.arange(1, 7), estimated_probabilities, 1, align="center", color="r")
    plt.show()

elif argv[1] == 'by_cat':
    I_list = [5, 30, 500, 1000000]
    for index, I in enumerate(I_list):
        original_probabilities = np.array([0.25, 0.15, 0.1, 0.1, 0.15, 0.25])
        r = np.random.choice(np.arange(1, 7), I, p=original_probabilities)

        prior = np.array([10, 5, 4, 4, 5, 10])
        estimated_probabilities = fitting.by_cat(r, prior)[1]
        
        plt.figure("Bayesian Cat")
        plt.subplot(421 + index * 2)
        plt.bar(np.arange(1, 7), original_probabilities, 1, align="center", color="blue")
        plt.subplot(422 + index * 2)
        plt.bar(np.arange(1, 7), estimated_probabilities, 1, align="center", color="r")
    plt.show()