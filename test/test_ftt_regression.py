import pytest

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nnu import ftt_regression as fttr

from nnu import gss_kernels as gssk
from nnu import points_generator as pgen
from nnu import gss_report_generator as gssgen


@pytest.mark.parametrize('input_f_spec', ['test', 'laplace_0'])
def test_tt_01(input_f_spec, plot_figures=False):
    ndim = 3
    sim_range = 4
    stretch = 1.0
    nx = 10000
    input_seed = 1917
    xs = pgen.generate_points(
        sim_range*1.0, nx, ndim, 'random', seed=input_seed)[0]

    if input_f_spec == 'test':
        def func(x): return (1/(1+x[:, 0]**2)) * \
            (1/(1+x[:, 1]**2))*(1/(1+x[:, 2]**2))

    else:
        genres = gssgen.generate_inputs_and_nodes(
            ndim=ndim,
            nsamples=nx,
            nnodes=nx,
            input_f_spec=input_f_spec,
            input_seed=input_seed,
            nsr_stretch=1.0,
        )
        func = genres[-1]
    ys = func(xs)

    nnodes = 2*int(pow(nx, 1.0/ndim))
    # kernel = 'bspline1'
    kernel = 'invquad'
    # kernel = 'lanczos'
    scale_mult = 4.0
    global_scale = 2*sim_range*stretch / nnodes
    knl_f = gssk.global_kernel_dict(global_scale * scale_mult)[kernel]
    nodes = np.linspace(-sim_range, sim_range, nnodes, endpoint=True)
    bf_vals = fttr.basis_function_values(xs, nodes, knl_f)

    tt_ranks = [1, 2, 3, 1]
    np.random.seed(31415)
    init_val = None
    tt_cores = fttr.init_tt(tt_ranks, nnodes, init_val=init_val)

    try_one_pass = True

    n_iter = 5
    start_time = time.time()
    for iter in range(n_iter):
        if try_one_pass:
            tt_cores, ys_fit = fttr.solve_one_pass(tt_cores, bf_vals, ys)
        else:
            for d in range(ndim):
                # print(f'd={d}')
                tt_cores, ys_fit = fttr.solve_for_dimension(
                    d=d, tt_cores=tt_cores, bf_vals=bf_vals, ys=ys)

        r2 = 1 - np.linalg.norm(ys_fit - ys)/np.linalg.norm(ys)
        print(f'iter = {iter} r2 = {r2}')
    end_time = time.time()
    print(f'time elapsed = {end_time - start_time:0.2f}')

    if plot_figures:
        plt.plot(ys, ys_fit, '.')
        plt.show()

        df = pd.DataFrame()
        for d in range(ndim):
            for i in range(tt_cores[d].shape[0]):
                for j in range(tt_cores[d].shape[2]):
                    f = fttr.get_function(d, i, j, tt_cores, knl_f, nodes)
                    label = f'{d}:{i}:{j}'
                    fs = f(xs)
                    df[label] = fs
                    plt.plot(xs[:, d], fs, '.',
                             markersize=1, label=label)
        plt.legend(loc='best')
        plt.show()

        df.to_csv('./results/ftt_res.csv')

    y_pred = fttr.predict(xs, tt_cores, knl_f, nodes)
    print(y_pred.shape)


@pytest.mark.parametrize('input_f_spec,ndim', [('doput_0', 5), ('laplace_0', 3), ('midrank_5', 11)])
def test_tt_02(input_f_spec, ndim, plot_figures=False):
    sim_range = 4
    stretch = 1.05
    nx = 10000
    input_seed = 1917

    # nnodes = 2*int(pow(nx, 1.0/ndim))
    nnodes = 11
    flat_rank = 5
    tt_ranks = [1] + [flat_rank]*(ndim-1) + [1]

    nodes = np.linspace(-sim_range*stretch, sim_range *
                        stretch, nnodes, endpoint=True)

    xs = pgen.generate_points(
        sim_range, nx, ndim, 'randomgrid', seed=input_seed, nodes1d=nodes)[0]

    if input_f_spec == 'test':
        def func(x): return (1/(1+x[:, 0]**2)) * \
            (1/(1+x[:, 1]**2))*(1/(1+x[:, 2]**2))

    else:
        genres = gssgen.generate_inputs_and_nodes(
            ndim=ndim,
            nsamples=nx,
            nnodes=nx,
            input_f_spec=input_f_spec,
            input_seed=input_seed,
        )
        func = genres[-1]
    ys = func(xs)

    # kernel = 'bspline1'
    kernel = 'invquad'
    # kernel = 'lanczos'
    # kernel = 'bspline2'
    scale_mult = 10.0

    global_scale = 2*sim_range*stretch / nnodes
    knl_f = gssk.global_kernel_dict(global_scale * scale_mult)[kernel]

    rel_tol = 1e-3
    max_iter = 25
    init_ttc_val = None
    learning_rate = 1.25

    # ridge regression seems to work better than lstsq
    onepass_params = {'use_ridge': True,
                      'ridge_alpha': 1e-12, 'print_iter_res': True, 'use_alpha_scaling': True,
                      'learning_rate': learning_rate}

    np.random.seed(input_seed)
    start_time = time.time()
    tt_cores, bf_vals, ys_fit, learn_mse, iter = fttr.fit(
        xs, ys, tt_ranks, nodes, knl_f,
        rel_tol=rel_tol, max_iter=max_iter, init_tt_cores_val=init_ttc_val, **onepass_params)
    end_time = time.time()
    fitting_time = end_time - start_time
    print(f'fitting_time = {fitting_time:0.2f}')

    if plot_figures:
        plt.plot(ys, ys_fit, '.')
        plt.show()

        df = pd.DataFrame()
        for d in range(ndim):
            label = f'x[{d}]'
            df[label] = xs[:, d]
            for i in range(tt_cores[d].shape[0]):
                for j in range(tt_cores[d].shape[2]):
                    f = fttr.get_function(d, i, j, tt_cores, knl_f, nodes)
                    label = f'{d}.{i}.{j}'
                    fs = f(xs)
                    df[label] = fs
                    plt.plot(xs[:, d], fs, '.',
                             markersize=1, label=label)
        plt.legend(loc='best')
        plt.show()

        df.to_csv('./results/ftt_res.csv')

    # figure out test error
    test_seed = input_seed * 2
    x_test = pgen.generate_points(
        sim_range*1.0, nx, ndim, 'random', seed=test_seed)[0]
    y_test_act = func(x_test)
    y_test_fit = fttr.predict(x_test, tt_cores, knl_f, nodes)
    test_mse = np.linalg.norm(y_test_fit - y_test_act) / \
        np.linalg.norm(y_test_act)
    print(f'learn mse = {learn_mse}')
    print(f'test mse = {test_mse}')
