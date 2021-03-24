import numpy as np
import matplotlib.pyplot as plt
from estimate_confounding_via_kernel_smoothing import estimate_confounding_via_kernel_smoothing, mat_vec_cov
import pandas as pd
from sklearn import linear_model


def calculate_vec_induced_spectral_measure(mat, vec):
    """
    :param mat: the matrix on which the measure is defines
    :param vec: the vector on which the measure is defines
    :return: the value of the measure and the matrix eigenvalues
    """
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    measure = np.matmul(vec, eigenvectors)**2
    measure = measure/np.sum(measure)
    return measure, eigenvalues


def plot_bar_graph(measure, values):
    """
    :param measure: the measure value for each eigenvalue
    :param values: the corresponding  eigenvalues
    :return: creates a bar plot that shows the measure distribution among the eigenvalues
    """
    plt.bar(values, measure, width=0.05)
    plt.xlim(0, 3.1)
    plt.xticks(values, rotation='vertical')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("taste_of_wine/winequality-red.csv", sep=';')
    X = data.to_numpy()
    Y = X[:, -1]
    X = X[:, :-2]  # removes the y and alcohol values to remove only y change to -1
    cxx = mat_vec_cov(X, X)
    scale = np.sqrt(np.diag(np.diag(cxx)))
    X = np.matmul(X, np.linalg.inv(scale))
    beta_est, eta_est = estimate_confounding_via_kernel_smoothing(X, Y)
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    a_est = reg.coef_
    cxx = mat_vec_cov(X, X)
    measure, values = calculate_vec_induced_spectral_measure(cxx, a_est)
    plot_bar_graph(measure, values)
    print('a estimate = {}'.format(a_est))
    print('beta estimation = {}'.format(beta_est))
