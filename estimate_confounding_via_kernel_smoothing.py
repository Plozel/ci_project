import numpy as np
from scipy.optimize import minimize

def estimate_confounding_via_kernel_smoothing(X, Y):
    """
    :param X: n x d matrix for the potential causes. Here d is the number of variables X_1,...,X_d and n is the number of samples
    :param Y: column vector of n instances of the target variable
    :return: 2-dimensional vector "parameters" with the confounding parameters (beta,eta) as in the paper, with beta being the relevant one since the
             estimation of eta performs rather bad for reasons that I don't completely understand yet.
    """

    d = X.shape[1]
    # Calculate covariance matrices
    cxx = np.cov(X)
    cxy = np.cov(X, Y)
    # closed-form solution for linear regression coefficients
    a_hat = np.matmul(np.linalg.inv(cxx), cxy)
    spectrumX, eigenvectors = np.linalg.eig(cxx)
    normed_spectrum = spectrumX / (max(spectrumX)-min(spectrumX))
    weights = np.matmul(np.T(a_hat), eigenvectors) ** 2
    weights = weights / np.sum(weights)
    smoothing_matrix = outer_with_kernel(normed_spectrum[:, None], normed_spectrum)
    smoothed_weights = np.matmul((smoothing_matrix, np.T(weights)))
    weights_causal = np.full(d, 1/d)
    parameters = optim_distance(d, spectrumX, weights_causal, smoothing_matrix, smoothed_weights)
    return parameters


def outer_with_kernel(value1, value2):
    """
    :param value1:
    :param value2:
    :return:
    """
    sigma = 0.2
    exp = np.exp(- ((value1 - value2) ** 2) / (2 * (sigma ** 2)))
    return exp


def optim_distance(d, spectrumX, weights_causal, smoothing_matrix, smoothed_weights):
    """

    :param d:
    :param spectrumX:
    :param weights_causal:
    :param smoothing_matrix:
    :param smoothed_weights:
    :return:
    """

    def get_distance(_lambda):
        """

        :param _lambda:
        :return:
        """
        g = np.full(d, 1/np.sqrt(d))
        T = np.diag(spectrumX) + _lambda[1] * np.matmul(g, np.T(g))
        _, eigenvectors_T = np.linalg.eig(T)
        weights_confounded = (spectrumX ** (-2)) * (np.matmul(g, eigenvectors_T) ** 2)
        weights_confounded = weights_confounded / np.sum(weights_confounded)
        weights_ideal = (1 - _lambda[0]) * weights_causal + _lambda[0] * weights_confounded
        smoothed_weights_ideal = np.matmul(smoothing_matrix, np.T(weights_ideal))
        dist = np.sum(np.abs(smoothed_weights - smoothed_weights_ideal))
        return (dist)

    params = minimize(get_distance, np.array([0, 0]), method="L-BFGS-B", bounds=[(0, 1), (0, 10)])
    return params


