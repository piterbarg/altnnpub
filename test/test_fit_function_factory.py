import pytest
import itertools
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from nnu import laplace_kernel
from nnu import fit_function_factory as fff


def helper_generate_01(kernel_type: fff.KernelType, general_nd: bool = True,
                       npts: int = 10, range_std: float = 1, stds: np.ndarray = None,
                       off_diag_correl: float = 0.0, shift=0.0, mean: np.ndarray = None, plot_figures=False, **kwargs):

    if general_nd:
        xs1d = np.linspace(-range_std, range_std, npts)

        x1, x2, x3 = np.meshgrid(xs1d, xs1d, xs1d)
        pos = np.stack((x1, x2, x3), axis=-1)
        xs = np.stack((x1.reshape(-1), x2.reshape(-1),
                       x3.reshape(-1)), axis=-1)

        cov = laplace_kernel.simple_covar_matrix(stds, off_diag_correl)

        fit_func = fff.generate_nd(
            kernel_type, cov, shift=shift, mean=mean, **kwargs)
        kernels = fit_func(xs)
        kern_grid = kernels.reshape(npts, npts, npts)
    else:
        fit_func = fff.generate_3d_grid(
            kernel_type, stds=stds, off_diag_correl=off_diag_correl, shift=shift, mean=mean, **kwargs)
        kern_grid, pos, *_ = fit_func(npts, range_std)

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


def helper_generate_smooth_rank_f_01(ndim: int, rank: int, plot_figures=False):

    tau = 5e0
    f = fff.generate_smooth_rank_f(ndim, rank, tau)
    x = 10*np.random.uniform(size=(10000, ndim))-5
    y = f(x)
    coord_x = 0
    coord_y = 1
    colour = 2

    if plot_figures:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:, coord_x], x[:, coord_y], y, c=x[:, colour],
                   cmap=cm.coolwarm, marker='.', label='y')
        plt.show()


def test_generate_random_01(plot_figures=False):
    ftfs = fff.generate_random(ndim=3, n_peaks=2, n_samples=4, seed=314)

    fit_functions = [f[1] for f in ftfs]

    range_std = 2
    npts = 20
    xs1d = np.linspace(-range_std, range_std, npts)

    x1, x2, x3 = np.meshgrid(xs1d, xs1d, xs1d)
    pos = np.stack((x1, x2, x3), axis=-1)
    xs = np.stack((x1.reshape(-1), x2.reshape(-1),
                   x3.reshape(-1)), axis=-1)

    for fit_func in fit_functions:
        kernels = fit_func(xs)
        kern_grid = kernels.reshape(npts, npts, npts)

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


@pytest.mark.parametrize('kernel_type, general_nd',
                         itertools.product([fff.KernelType.Gs, fff.KernelType.Lp, fff.KernelType.LpM], [True, False]))
def test_all_kernels(kernel_type, general_nd):
    helper_generate_01(kernel_type, general_nd,
                       npts=50, range_std=1.0, stds=[0.5, 1, 1.5], off_diag_correl=0.75, mean=[0.25, 0.5, 0.75])


def test_generate_01():
    helper_generate_01(
        fff.KernelType.LpM,
        general_nd=False,
        npts=50,
        range_std=2.0,
        stds=[0.5, 1, 1.5],
        off_diag_correl=0.75,
        shift=0.1,
        means=[[1, 1, 0], [-1, -1, 0], [1, -1, 0]],
        cov_multipliers=[0.8, 1.0, 1.25],
        mix_weights=[0.3, 0.3, 0.4]
    )


@pytest.mark.parametrize('ndim', range(1, 6))
def test_generate_smooth_rank_f_01(ndim):
    helper_generate_smooth_rank_f_01(ndim=ndim, rank=ndim//3)
    assert True


@pytest.mark.parametrize('ndim', range(6))
def test_generate_down_out_put(ndim: int, plot_figures=False):

    sim_range = 4.0
    x = 2*sim_range*np.random.uniform(size=(10000, 5))-sim_range
    f = fff.generate_down_out_put(ndim, sim_range)
    y = f(x)
    coord_x = 0
    coord_y = 1
    colour = 2

    if plot_figures:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:, coord_x], x[:, coord_y], y, c=x[:, colour],
                   cmap=cm.coolwarm, marker='.', label='y')
        plt.show()

    assert True
