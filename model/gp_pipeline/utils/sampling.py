import numpy as np
from scipy.stats import qmc

def create_lhs_samples(n_dim, n_points, seed=None):
    sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
    return sampler.random(n=n_points)

def create_random_samples(n_dim, n_points, seed=None):
    sampler = np.random.default_rng(seed)
    return sampler.random((n_points, n_dim))