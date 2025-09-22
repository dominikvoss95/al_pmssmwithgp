import numpy as np
from scipy.stats import multivariate_normal

def create_two_gaussian_hyperspheres(n_dim):
    '''Function to create two gaussian hyperspheres'''
    mean1 = [0.75] * n_dim
    cov1 = np.diag([0.03] * n_dim)

    mean2 = [0.25] * n_dim
    cov2 = np.diag([0.03] * n_dim)

    def truth_fn(X):
        val1 = multivariate_normal.pdf(X, mean=mean1, cov=cov1)
        val2 = multivariate_normal.pdf(X, mean=mean2, cov=cov2)
        return val1 + val2 + 0.1  # Added constant offset for smooth background

    return truth_fn

def create_two_different_circles(n_dim):
    '''Function to create two circles of different sizes'''
    means = []
    covs = []
    gaussians = []

    mean1 = [0.8] * n_dim
    cov1 = np.diag([0.03] * n_dim)
    scale1 = 0.05

    mean2 = [0.48] * n_dim  
    cov2 = np.diag([0.03] * n_dim) 
    scale2 = 0.05

    means.append(mean1)
    covs.append(cov1)

    means.append(mean2)
    covs.append(cov2)

    gaussians.append(lambda X, mean=mean1, cov=cov1: scale1 * np.atleast_1d(multivariate_normal.pdf(X, mean=mean1, cov=cov1)))
    gaussians.append(lambda X, mean=mean2, cov=cov2: scale2 * np.atleast_1d(multivariate_normal.pdf(X, mean=mean2, cov=cov2)))

    # return lambda X: sum(gaussian(X) for gaussian in gaussians)
    return lambda X: sum(gaussian(X) for gaussian in gaussians) + 0.1
