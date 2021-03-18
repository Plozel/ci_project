import numpy as np
from numpy import random
from math import sqrt
import matplotlib.pyplot as plt
from estimate_confounding_via_kernel_smoothing import estimate_confounding_via_kernel_smoothing


def simulation(d, sample_size, runs):
    """
    :param d: dimension
    :param sample_size: sample
    :param runs: number of runs
    :return: scatter plot that shows the relation between true and estimated confounding strength beta.
             The number of points is given by the parameter 'runs'
    """

    beta = []
    eta = []
    beta_est = []
    eta_est = []

    for i in range(runs):
        # randomly draw model parameters

        # a's sphere radius
        r_a = random.uniform(0, 1)
        # b's sphere radius
        r_b = random.uniform(0, 1)
        # strength of influence of confounder on target variable Y
        c = random.uniform(0, 1)

        # draw random vectors a and b
        a = random.normal(0, 1, d)
        b = random.normal(0, 1, d)
        a = a / sqrt(sum(a ** 2)) * r_a
        b = b / sqrt(sum(b ** 2)) * r_b

        # generate samples of the noise vector E
        E = random.normal(0, 1, (sample_size, d))
        # Refer G as a parameter
        G = random.normal(0, 1, (d, d))
        E = np.matmul(E, G)

        # generate samples of the confounder Z and the noise term NY for the target variable Y
        Z = random.normal(0, 1, sample_size)
        F = random.normal(0, 1, sample_size)

        # compute X and Y via linear structural equations
        X = E + np.outer(Z, b)
        Y = c * Z + np.matmul(X, a) + F

        # compute confounding parameters
        SigmaEE = np.matmul(np.transpose(G), G)
        SigmaXX = SigmaEE + np.outer(b, b)
        confounding_vector = c * np.matmul(np.linalg.inv(SigmaXX), b)
        sq_length_cv = np.sum(confounding_vector ** 2)
        beta.append(sq_length_cv / (r_a ** 2 + sq_length_cv))
        eta.append(r_b ** 2)

        # estimate both confounding parameters
        parameters = estimate_confounding_via_kernel_smoothing(X, Y)
        beta_est.append(parameters[0])
        eta_est.append(parameters[1])

    return beta_est, beta


if __name__ == '__main__':
    d = 20
    sample_size = 10000
    runs = 1000

    beta_est, beta = simulation(d, sample_size, runs)
    fig = plt.figure()
    plt.title('d = {} n = {}'.format(d, sample_size))
    s = plt.scatter(beta, beta_est, s=10, marker='*')
    plt.show()
