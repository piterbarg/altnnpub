import numpy as np
import pandas as pd
from enum import Enum
import tensorflow as tf
from matplotlib import cm
import matplotlib.pyplot as plt

from nnu import gauss_kernel, laplace_kernel, option_pricers


class KernelType(Enum):
    Gs = 0
    Lp = 1
    LpM = 2


def generate_nd(
        kernel_type: KernelType, cov: np.ndarray, **kwargs):
    '''
    Returns a function xs->f(xs) that can be used to calculate a function we are trying to fit
    a uniform-ish interface over a few functions of interest
    '''

    if kernel_type == KernelType.Gs:
        return lambda xs: gauss_kernel.gauss_kernel_values(xs, cov, mean=kwargs.get('mean'))

    if kernel_type == KernelType.Lp:

        s = kwargs.get('shift', 0.0)
        return lambda xs: laplace_kernel.laplace_kernel_values(xs, cov, mean=kwargs.get('mean'), shift=s)

    if kernel_type == KernelType.LpM:
        cov_multipliers = kwargs.pop('cov_multipliers', None)
        means = kwargs.pop('means', None)
        weights = kwargs.pop('mix_weights', None)

        if cov_multipliers is None and means is None and weights is None:
            # a pre-canned test case
            cov_multipliers = [0.8, 1.25]
            means = [[1.0]*cov.shape[0], [-1.0]*cov.shape[0]]
            weights = [0.75, 0.25]

        # get rid of 'mean' in kwargs if it is there
        kwargs.pop('mean', None)

        kernels = [generate_nd(KernelType.Lp, cov=cov*cm, mean=m, **kwargs)
                   for cm, m in zip(cov_multipliers, means)]

        return lambda xs: sum([w*k(xs) for w, k in zip(weights, kernels)])

    raise ValueError(
        f'Unsupported kernel type: {kernel_type}')


def generate_3d_grid(
        kernel_type: KernelType, **kwargs):
    '''
    Returns a function _->f() that can be used to calculate a function we are trying to fit
    a uniform-ish interface over a few functions of interest. on a three-d grid
    '''

    cov_multipliers = kwargs.pop('cov_multipliers', None)
    means = kwargs.pop('means', None)
    weights = kwargs.pop('mix_weights', None)

    if kernel_type == KernelType.Gs:
        # get rid of 'shift' in kwargs if it is there
        kwargs.pop('shift', None)
        return lambda npts, range_std: gauss_kernel.generate_test_kernel_3d(npts, range_std, **kwargs)

    if kernel_type == KernelType.Lp:
        return lambda npts, range_std: laplace_kernel.generate_test_kernel_on_mesh_3d(npts, range_std, **kwargs)[0:2]

    if kernel_type == KernelType.LpM:
        # get rid of 'mean', 'stds' in kwargs if it is there
        kwargs.pop('mean', None)
        stds = kwargs.pop('stds', None)

        if cov_multipliers is None and means is None and weights is None:
            # a pre-canned test case
            cov_multipliers = [0.8, 1.25]
            means = [[1.0]*len(stds), [-1.0]*len(stds)]
            weights = [0.75, 0.25]

        kernels = [generate_3d_grid(KernelType.Lp, stds=np.array(stds)*np.sqrt(cm), mean=m, **kwargs)
                   for cm, m in zip(cov_multipliers, means)]

        def fit_func(npts, range_std):
            evaluated_kernels = [k(npts, range_std) for k in kernels]
            return (sum([w*k[0] for w, k in zip(weights, evaluated_kernels)]),
                    evaluated_kernels[0][1])

        return fit_func

    raise ValueError(
        f'Unsupported kernel_type: {kernel_type}')


def generate_random(
        ndim: int, n_peaks: int, n_samples: int = 10, seed: int = 314):
    '''
    Generate random functions, useful for large-scale testing 
    '''
    ftfs = []
    np.random.seed(seed)
    for _ in range(n_samples):
        stds = np.random.uniform(low=0.25, high=2.0, size=ndim)
        off_diag_correl = np.random.uniform(low=0, high=0.5)
        laplace_shift = np.random.uniform(low=0.1, high=0.9)
        means = np.random.uniform(low=-2.0, high=2.0, size=(n_peaks, ndim))
        cov_multipliers = np.random.uniform(low=0.1, high=0.9, size=n_peaks)
        mix_weights = np.random.uniform(low=0.1, high=0.9, size=n_peaks)

        kernel_type = KernelType.LpM
        covar_matr = laplace_kernel.simple_covar_matrix(stds, off_diag_correl)
        function_to_fit = generate_nd(
            kernel_type, covar_matr, shift=laplace_shift,
            means=means,
            cov_multipliers=cov_multipliers,
            mix_weights=mix_weights,
        )

        function_params = {
            'stds': stds,
            'off_diag_correl': off_diag_correl,
            'laplace_shift': laplace_shift,
            'means': means,
            'cov_multipliers': cov_multipliers,
            'mix_weights': mix_weights,
            'kernel_type': kernel_type,
        }

        ftfs.append((function_params, function_to_fit))
    return ftfs


def generate_smooth_rank_f(
    ndim: int, rank: int, tau: float, dim_weights=None, skip_dims=0,
):

    # from https://arxiv.org/pdf/2006.16038.pdf
    def soft_sort_prillo(x, tau):
        s = tf.expand_dims(tf.constant(x), -1)
        s_sorted = tf.sort(s, direction='DESCENDING', axis=1)
        pairwise_distances = - \
            tf.abs(tf.transpose(s, perm=[0, 2, 1]) - s_sorted)
        P_hat = tf.nn.softmax(pairwise_distances / tau, -1)
        return P_hat.numpy()

    def softsort_np(x, tau=5e0, rank=0):
        sr = soft_sort_prillo(x, tau=tau)
        return np.matmul(sr, np.expand_dims(x, -1))[:, rank, 0]

    if dim_weights is None:
        dim_weights = np.ones(ndim)
    return lambda x: softsort_np((x*dim_weights)[:, skip_dims:ndim-skip_dims], tau, rank=rank - skip_dims)


def generate_down_out_put(
    ndim: int,
    sim_range: float = 4.0,
    spot: float = 100,
    strike: float = 100,
    strike_range_rel=(0.5, 1.5),
    barrier: float = 50,
    barrier_range_rel=(0.5, 1.5),
    vol: float = 0.2,
    vol_range_rel=(0.25, 2.5),
    expiry=1,
    expiry_range_rel=(0.5, 2),
    rate=0.05,
    rate_range_rel=(0.0, 2)
):
    def down_out_put_val(x):
        '''
        x is an array of nsamples x ndim where ndim <= 5. If ndim < 5 we will use
        x as inputs for the pricer parameters in the order they are mentioned
        in the signature of the outer function ie (strike, barrier, vol, expiry, rate)

        x comes from [-sim_range,sim_range] domain for each dimension. These need to be rescaled

        '''

        npts = x.shape[0]

        # start with flat values
        ks = np.ones(npts)*strike/spot
        bs = np.ones(npts)*barrier/spot
        vols = np.ones(npts)*vol
        ts = np.ones(npts)*expiry
        rs = np.ones(npts)*rate

        # now on [0,1]
        x0 = x / (2*sim_range) + 0.5

        # now use x0 for the first ndim dimensions
        if ndim >= 1:
            ks = strike/spot * \
                (strike_range_rel[0] + (strike_range_rel[1] -
                                        strike_range_rel[0])*x0[:, 0])
        if ndim >= 2:
            bs = barrier/spot * \
                (barrier_range_rel[0] + (barrier_range_rel[1] -
                                         barrier_range_rel[0])*x0[:, 1])
        if ndim >= 3:
            vols = vol * \
                (vol_range_rel[0] + (vol_range_rel[1] -
                                     vol_range_rel[0])*x0[:, 2])
        if ndim >= 4:
            ts = expiry * \
                (expiry_range_rel[0] + (expiry_range_rel[1] -
                                        expiry_range_rel[0])*x0[:, 3])
        if ndim >= 5:
            rs = rate * \
                (rate_range_rel[0] + (rate_range_rel[1] -
                                      rate_range_rel[0])*x0[:, 4])

        return option_pricers.down_out_put(bs=bs, ks=ks, ts=ts, vols=vols, rs=rs)*spot

    return down_out_put_val
