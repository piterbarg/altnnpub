import pytest
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from nnu import gauss_kernel


def helper_gauss_kernel_3d_01(npts=10, off_diag_correl=0.0, mean: np.ndarray = None, plot_figures=False):
    kernel, pos = gauss_kernel.generate_test_kernel_3d(
        npts=npts, range_std=2.0, stds=np.array([0.5, 1, 1.5]), off_diag_correl=off_diag_correl, mean=mean)
    ksum = np.sum(kernel)
    print(f'sum={ksum}')

    if plot_figures:
        # plot in x1,x2 direction, for  a few indices at the third direction
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x3_idx_all = [npts//4, npts//2, 3*npts//4]
        for x3_idx in x3_idx_all:
            x1 = pos[:, :, x3_idx, 0]
            x2 = pos[:, :, x3_idx, 1]
            kernel2d = kernel[:, :, x3_idx]

            ax.scatter(x1.reshape(npts*npts),
                       x2.reshape(npts*npts),
                       kernel2d.reshape(npts*npts), cmap=cm.coolwarm)

        plt.show()
    return ksum


def helper_gauss_kernel_values_01(
        npts: int = 10, range_std: float = 1,
        stds: np.ndarray = None, off_diag_correl: float = 0.0, mean: np.ndarray = None, plot_figures=False):

    from nnu.laplace_kernel import simple_covar_matrix
    cov = simple_covar_matrix(stds, off_diag_correl)
    xs1d = np.linspace(-range_std, range_std, npts)

    x1, x2, x3 = np.meshgrid(xs1d, xs1d, xs1d)
    pos = np.stack((x1, x2, x3), axis=-1)
    xs = np.stack((x1.reshape(-1), x2.reshape(-1),
                   x3.reshape(-1)), axis=-1)

    kernels = gauss_kernel.gauss_kernel_values(xs, cov, mean=mean)
    kern_grid = kernels.reshape(npts, npts, npts)
    ksum = np.sum(kernels)
    print(f'sum={ksum}')

    if plot_figures:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x3_idx_all = [npts//4, npts//2, 3*npts//4]
        for x3_idx in x3_idx_all:
            x1 = pos[:, :, x3_idx, 0]
            x2 = pos[:, :, x3_idx, 1]
            kernel2d = kern_grid[:, :, x3_idx]

            ax.scatter(x1.reshape(npts*npts),
                       x2.reshape(npts*npts),
                       kernel2d.reshape(npts*npts),
                       cmap=cm.coolwarm, marker='.')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
    return ksum


def test_gauss_kernel_3d_01():
    out = helper_gauss_kernel_3d_01(
        10, 0.5, mean=[1, 2, 3])
    assert pytest.approx(out) == pytest.approx(279.6086309528281)


def test_gauss_kernel_values_01():
    out = helper_gauss_kernel_values_01(
        50, 1.0, stds=[0.5, 1, 1.5], off_diag_correl=0.75, mean=[0.25, 0.5, 0.75])
    assert pytest.approx(out) == pytest.approx(2647.404040991063)
