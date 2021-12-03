import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def laplace_kernel_values(
        xs: np.ndarray, cov: np.ndarray, shift=0, mean: np.ndarray = None):
    '''
    This is not really a multivariate Laplace PDF, but close enough
    v(x) = exp(-sqrt(sum xs[i]^2)) for cov = I
    or
    v(x) = exp(-sqrt(x'C^{-1} x + shift)) for cov = C

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

    return np.exp(-np.sqrt(ds + shift))


def simple_covar_matrix(stds: np.ndarray = None, off_diag_correl: float = 0.0):
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

    return cov


def generate_test_kernel_on_mesh_3d(
        npts: int = 10, range_std: float = 1, stds: np.ndarray = None,
        off_diag_correl: float = 0.0, shift: float = 0.0, mean: np.ndarray = None):

    cov = simple_covar_matrix(stds, off_diag_correl)
    xs1d = np.linspace(-range_std, range_std, npts)

    x1, x2, x3 = np.meshgrid(xs1d, xs1d, xs1d)
    pos = np.stack((x1, x2, x3), axis=-1)
    xs = np.stack((x1.reshape(-1), x2.reshape(-1), x3.reshape(-1)), axis=-1)

    kernels = laplace_kernel_values(xs, cov, shift=shift, mean=mean)
    kern_grid = kernels.reshape(npts, npts, npts)
    return kern_grid, pos, kernels, xs
