import pytest
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nnu import laplace_kernel


@pytest.mark.parametrize('npts,off_diag_correl,shift,mean',
                         [(50, 0.5, 1.0, [1, 2, 3])])
def test_laplace_kernel_3d_01(npts, off_diag_correl, shift, mean, plot_figures=False):
    kern_grid, pos, *_ = laplace_kernel.generate_test_kernel_on_mesh_3d(
        npts=npts, range_std=2.0, stds=np.array([0.5, 1, 1.5]),
        off_diag_correl=off_diag_correl, shift=shift, mean=mean)
    print(f'sum={np.sum(kern_grid)}')

    if plot_figures:
        # plot in x1,x2 direction, for  a few indices at the third direction
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
