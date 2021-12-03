import math
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def gauss_kernel_values(xs: np.ndarray, cov: np.ndarray, mean: np.ndarray = None):
    '''
    Vectorized Gaussian Kernel
    v(x) ~ exp(-x'C^{-1} x/2) for cov = C

    xs.shape = (n_obs, n_dim)
    cov is the covariance matrix n_dim x n_dim
    returns 1d array of n_dim
    '''

    if mean is None:
        mean = np.zeros(xs.shape[1])

    xs = xs - mean

    # calculating a vectorized version of x' cov^{-1} x
    Cinv = np.linalg.inv(cov)
    ds = np.sum(xs * (xs @ Cinv), axis=1)
    # alegedly this is faster: numpy.einsum("ij,ij->i", xs, xs @ Cinv)

    dim = xs.shape[1]
    detC = np.linalg.det(cov)
    nrm = (2*math.pi)**(dim/2.0)*math.sqrt(detC)
    return np.exp(-ds)/nrm


def discretized_gauss_kernel_values_3d(xs: np.ndarray, cov: np.ndarray, mean: np.ndarray = None):
    rv = multivariate_normal(mean=mean, cov=cov)
    x1, x2, x3 = np.meshgrid(xs, xs, xs)
    pos = np.stack((x1, x2, x3), axis=-1)
    vals = 1e2*rv.pdf(pos)
    return vals, pos


def generate_test_kernel_3d(
        npts: int = 10, range_std: float = 1, stds: np.ndarray = None, off_diag_correl: float = 0.0, mean: np.ndarray = None):

    if stds is None:
        stds = np.array([1.0, 1.0, 1.0])

    dim = len(stds)

    cov = np.zeros((dim, dim))
    for i in np.arange(dim):
        cov[i, i] = stds[i]**2

    for i in np.arange(dim):
        for j in np.arange(i):
            cov[i, j] = off_diag_correl * np.sqrt(cov[i, i] * cov[j, j])
            cov[j, i] = cov[i, j]

    xs = np.linspace(-range_std, range_std, npts)
    kernel, pos = discretized_gauss_kernel_values_3d(xs, cov, mean=mean)
    return kernel, pos
