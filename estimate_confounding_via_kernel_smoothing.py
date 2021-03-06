import numpy as np
from scipy.optimize import minimize
from sklearn import linear_model


def kernel(X):
    """
    :param X: data to be refitted
    :return: the refitted data
    """
    return X**2


def estimate_confounding_via_kernel_smoothing(X, Y, lamda=0, norm=False):
    """
    :param X: n x d matrix for the potential causes. Here d is the number of variables X_1,...,X_d and n is the number of samples
    :param Y: column vector of n instances of the target variable
    :param lamda: the lambda parameter for the regularization.
    :param norm: a bool variable stating either to normalize X or not
    :return: 2-dimensional vector "parameters" with the confounding parameters (beta,eta) as in the paper, with beta being the relevant one since the
             estimation of eta performs rather bad for reasons that I don't completely understand yet.
    """
    if norm:
        # normalizing X using the inverse covariance matrix
        cxx = mat_vec_cov(X, X)
        scale = np.sqrt(np.diag(np.diag(cxx)))
        X = np.matmul(X, np.linalg.inv(scale))
    reg = linear_model.Ridge(alpha=lamda)
    d = X.shape[1]
    # Calculate covariance matrices
    cxx = mat_vec_cov(X, X)
    # linear regression
    reg.fit(X, Y)
    a_hat = np.array(reg.coef_)
    spectrumX, eigenvectors = np.linalg.eig(cxx)
    normed_spectrum = spectrumX / (max(spectrumX)-min(spectrumX))
    weights = np.matmul(np.transpose(a_hat), eigenvectors) ** 2
    weights = weights / np.sum(weights)
    smoothing_matrix = outer_with_kernel(normed_spectrum[:, None], normed_spectrum)
    smoothed_weights = np.matmul(smoothing_matrix, np.transpose(weights))
    weights_causal = np.full(d, 1/d)
    parameters = optim_distance(d, spectrumX, weights_causal, smoothing_matrix, smoothed_weights)
    return parameters


def outer_with_kernel(value1, value2):
    """
    :param value1: broad of the spectrum to squared matrix
    :param value2: the spectrum as a vector
    :return: the result of the kernel
    """
    sigma = 0.2
    exp = np.exp(- ((value1 - value2) ** 2) / (2 * (sigma ** 2)))
    return exp


def optim_distance(d, spectrumX, weights_causal, smoothing_matrix, smoothed_weights):
    """
    :param d: the number of dimensions of X
    :param spectrumX: the eigenvalues of cxx as a vector
    :param weights_causal: a vector of length d of 1/d estimating the weights of the causal a on the spectrum
    :param smoothing_matrix: the K matrix from the paper
    :param smoothed_weights: the real weights multiplied by K
    :return: an estimation of beta
    """

    def get_distance(_lambda):
        """
        :param _lambda: the function to minimize
        :return: the optimal estimations of beta and eta
        """
        g = np.full(d, 1/np.sqrt(d))
        T = np.diag(spectrumX) + _lambda[1] * np.outer(g, g)
        _, eigenvectors_T = np.linalg.eig(T)
        weights_confounded = (spectrumX ** (-2)) * (np.matmul(g, eigenvectors_T) ** 2)
        weights_confounded = weights_confounded / np.sum(weights_confounded)
        weights_ideal = (1 - _lambda[0]) * weights_causal + _lambda[0] * weights_confounded
        smoothed_weights_ideal = np.matmul(smoothing_matrix, np.transpose(weights_ideal))
        dist = np.sum(np.abs(smoothed_weights - smoothed_weights_ideal))
        return dist

    params = minimize(get_distance, np.array([0, 0]), method="L-BFGS-B", bounds=[(0, 1), (0, 10)])
    return params['x']


def mat_vec_cov(X, y):
    """
    :param X: np array nxm
    :param y: np array nxl
    :return: the covariance of the arrays
    """
    X = X - np.mean(X, axis=0)
    y = y - np.mean(y, axis=0)
    cov = np.matmul(np.transpose(X), y)
    n = X.shape[0]
    cov = cov/(n-1)
    return cov
