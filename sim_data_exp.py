import numpy as np
from numpy import random
from math import sqrt


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

        # length of vector a
        r_a = random.uniform(0, 1)
        # length of vector b
        r_b = random.uniform(0, 1)
        # strength of influence of confounder on target variable Y
        c = random.uniform(0, 1)
        # draw random vectors a and b

        a = random.normal(0, 1, d)
        b = random.normal(0, 1, d)
        a = a / sqrt(sum(a ^ 2)) * r_a
        b = b / sqrt(sum(b ^ 2)) * r_b

        # generate samples of the noise vector E
        E = random.normal(0, 1, (sample_size, d))
        random_matrix = random.normal(0, 1, (d, d))
        E = np.matmul(E, random_matrix)

        # generate samples of the confounder Z and the noise term NY for the target variable Y
        Z = random.normal(0, 1, sample_size)
        NY = random.normal(0, 1, sample_size)

        # compute X and Y via linear structural equations
        X = E + np.matmiul(Z, np.T(b))
        Y = c * Z + np.matmul(X, a) + NY

        # compute confounding parameters
        SigmaEE = np.matmul(np.T(random_matrix), random_matrix)
        SigmaXX = SigmaEE + np.matmul(b, np.T(b))
        confounding_vector = X * np.matmul(np.linalg.inv(SigmaXX), b)
        sq_length_cv = np.sum(confounding_vector ** 2)
        beta.append(sq_length_cv / (r_a ** 2 + sq_length_cv))
        eta.append(r_b ** 2)

        # estimate both confounding parameters
        parameters = estimate_confounding_via_kernel_smoothing(X, Y)
        beta_est[i] = parameters[1]
        eta_est[i] = parameters[2]

}

# TODO: plot