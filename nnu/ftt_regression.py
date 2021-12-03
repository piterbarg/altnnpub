import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import numpy.polynomial.chebyshev as cheb

from nnu import nn_utils
from nnu import gss_kernels as gssk
from nnu import points_generator as pgen
from nnu import gss_report_generator as gssgen


def basis_function_values(xs, nodes, kernel_f):
    '''
    Calculate values of basis functions at xs for each dimension
    xs: KxD learning set
    nodes: 1D nodes for each dimension, vector of length N. Assumed here same for all dims, could be relaxed later
    kernel_f: a kernel function as eg returned by gss_model_factory.global_kernel_dict()
    '''

    # try chebychev
    use_cheb = False

    if use_cheb:
        # hardcode chebyshev for testing
        return basis_function_values_cheb(xs, nodes)

    ndim = xs.shape[1]
    bf_vals = []
    for n in range(ndim):
        bf_vals_n = kernel_f(xs[:, n:n+1] - np.expand_dims(nodes, 0)).numpy()
        bf_vals.append(bf_vals_n)

    return bf_vals


def basis_function_values_cheb(xs, nodes):
    '''
    Calculate values of Chebyshev basis functions at xs for each dimension
    xs: KxD learning set
    nodes: we just use len(nodes) as the number of basis functions. and nodes[-1] to scale things
    '''
    sim_range = nodes[-1]
    nnodes = len(nodes)
    ndim = xs.shape[1]
    bf_vals = []
    for n in range(ndim):
        bf_vals_n = cheb.chebvander(xs[:, n]/sim_range, nnodes-1)
        bf_vals.append(bf_vals_n)

    return bf_vals


def init_tt(tt_ranks, nnodes, init_val=None):
    '''
    Create a list of TT cores of the right dimensions

    tt_ranks: vector [1,r_1,...r_{D-1},1]
    nnodes: how many nodes/basis functions per dimension. assumed the same for all dim
    init_val: the value to initialize each core to. A double or None in which case we do random init
    '''
    if type(init_val) is list:
        # not much to do
        return copy.deepcopy(init_val)

    ndim = len(tt_ranks) - 1
    tt_cores = []
    for n in range(1, ndim+1):
        if init_val is None:
            core_n = np.random.uniform(low=0.0, high=1.0,
                                       size=(tt_ranks[n-1], nnodes, tt_ranks[n]))
        else:
            core_n = init_val * np.ones((tt_ranks[n-1], nnodes, tt_ranks[n]))
        tt_cores.append(core_n)

    return tt_cores


def solve_for_dimension(d, tt_cores, bf_vals, ys):
    '''
    Apply least squares to dimension d, d=0,...,ndim-1
    Perhaps we can optimize stuff here later but now a bit of a brute force

    d: which dimension to apply LS to
    tt_cores: TT cores before the least squares
    bf_vals: basis function values at xs as returned by basis_function_values()
    ys: what we are fitting to

    returns: tt_cores updated with the new tt_cores[d]. Possibly modifies the input too
    '''

    ndim = len(tt_cores)
    nx = bf_vals[0].shape[0]

    rleft = tt_cores[d].shape[0]
    nnodes = tt_cores[d].shape[1]
    rright = tt_cores[d].shape[2]

    mleft = np.ones((nx, 1, 1))
    for d1 in range(d):
        # print(d1)
        g = np.dot(bf_vals[d1], tt_cores[d1])
        mleft = np.matmul(mleft, g)

    mright = np.ones((nx, 1, 1))
    for d1 in range(ndim-1, d, -1):
        # print(d1)
        g = np.dot(bf_vals[d1], tt_cores[d1])
        mright = np.matmul(g, mright)

    M = np.transpose(np.matmul(mright, mleft), (0, 2, 1)).reshape(nx, -1)
    A = np.matmul(M.reshape(nx, -1, 1), bf_vals[d].reshape(nx, 1, -1))
    Af = A.reshape(nx, -1)
    L = np.linalg.lstsq(Af, ys, rcond=None)[0]

    # we were trying the options, option 3 is the one that works!
    # options = [0, 1, 2, 3, 4, 5]
    # options[0] = np.transpose(L.reshape(nnodes, rleft, rright), (1, 0, 2))
    # options[1] = np.transpose(L.reshape(nnodes, rright, rleft), (2, 0, 1))
    # options[2] = np.transpose(L.reshape(rleft, nnodes, rright), (0, 1, 2))
    # options[3] = np.transpose(L.reshape(rleft, rright, nnodes), (0, 2, 1))
    # options[4] = np.transpose(L.reshape(rright, rleft, nnodes), (1, 2, 0))
    # options[5] = np.transpose(L.reshape(rright, nnodes, rleft), (2, 1, 0))

    # for n, opt in enumerate(options):
    #     print(f'option {n}:')
    #     tt_cores[d] = opt

    tt_cores[d] = np.transpose(L.reshape(rleft, rright, nnodes), (0, 2, 1))

    # our projection
    ys_fit = Af @ L

    # # check we got the indexing right
    # g = np.dot(bf_vals[d], tt_cores[d])
    # mleft2 = np.matmul(mleft, g)
    # ys_fit_2 = np.squeeze(np.matmul(mleft2, mright))
    # print(f'fit vs fit2: {np.linalg.norm(ys_fit - ys_fit_2)}')

    return tt_cores, ys_fit  # , y_fit_2

    # return mleft, mright


def solve_one_pass(
        tt_cores, bf_vals, ys, use_ridge=True, ridge_alpha=1e-4, use_alpha_scaling=True, iter_n=0, learning_rate=1.0):
    '''
    Apply least squares to all dimensions dimension d, d=0,...,ndim-1
    once. This is an attempt to optimize solve_for_dimension(...)
    by storing and reusing intermediat ecalculations

    PS Optimization did not do anything as most of the time is spent in np.linalg.lstsq
    as shown by the profiler:
    python -m cProfile -m nnu.ftt_regression > profile_res.txt

    tt_cores: TT cores before the least squares
    bf_vals: basis function values at xs as returned by basis_function_values()
    ys: what we are fitting to

    learning_rate       : move in the direction of the optimum by lr amount (lr=1 id full move)

    returns: updated tt_cores. Possibly modifies the input too
    '''

    # static data
    ndim = len(tt_cores)
    nx = bf_vals[0].shape[0]
    nnodes = tt_cores[0].shape[1]

    # inner function to solve for current d
    def solve_for_dimension_(mleft_d, bf_vals_d, mright_d, tt_cores_d):
        rleft = mleft_d.shape[2]
        rright = mright_d.shape[1]
        M = np.transpose(np.matmul(mright_d, mleft_d),
                         (0, 2, 1)).reshape(nx, -1, 1)
        A = np.matmul(M, bf_vals_d.reshape(nx, 1, -1))
        Af = A.reshape(nx, -1)

        if use_ridge:
            # scale the alpha to dimensionless units
            alpha_scaling = 1.0
            if use_alpha_scaling:
                alpha_scaling = np.linalg.norm(np.sum(Af, axis=1))**2/nnodes
                # alpha_scaling /= 2**(iter_n/2.0)

            # Lasso is too slow
            # regr = Lasso(alpha=alpha_scaling *
            #             ridge_alpha, fit_intercept=False, selection='random')
            regr = Ridge(alpha=alpha_scaling *
                         ridge_alpha, fit_intercept=False)
            regr_res = regr.fit(Af, ys)
            L = regr_res.coef_
        else:
            L = np.linalg.lstsq(Af, ys, rcond=None)[0]

        tt_cores_d_tgt = np.transpose(
            L.reshape(rleft, rright, nnodes), (0, 2, 1))
        if abs(learning_rate - 1.0) > 1e-4:
            tt_cores_d_tgt = learning_rate * \
                tt_cores_d_tgt + (1.0-learning_rate)*tt_cores_d

        return tt_cores_d_tgt

    # rleft = tt_cores[d].shape[0]
    # rright = tt_cores[d].shape[2]

    # This  will be updated by the inner function as we go through all d's
    mleft = np.ones((nx, 1, 1))

    # Precompute and store these. We put copies for each d on the stack
    mright_stack = []
    mright = np.ones((nx, 1, 1))

    mright_stack.append(mright.copy())
    for d1 in range(ndim-1, 0, -1):
        # print(d1)
        g = np.dot(bf_vals[d1], tt_cores[d1])
        mright = np.matmul(g, mright)
        mright_stack.append(mright.copy())

    for d in range(ndim):
        tt_cores[d] = solve_for_dimension_(
            mleft, bf_vals[d], mright_stack.pop(), tt_cores[d])

        # print(d1)
        g = np.dot(bf_vals[d], tt_cores[d])
        mleft = np.matmul(mleft, g)

    # ys_fit = np.squeeze(np.matmul(mleft, mright_stack.pop()))
    ys_fit = np.squeeze(mleft)
    return tt_cores, ys_fit


def fit(xs, ys, tt_ranks, nodes, kernel_f,
        rel_tol=1e-3, max_iter=10, init_tt_cores_val=None, **kwargs):
    '''
    Fit functional TT to data

    xs                  : K x D learning xs
    ys                  : K x 1 learning ys
    tt_ranks            : vector of tt ranks
    nodes               : one-dim vector of nodes (kernel centres), the same for each dimension
    kernel_f            : kernel (basis) function that we shift
    rel_tol             : stopping criteria for iterations: stop when relative improvement in tt core coefs is below rel_tol
    max_iter            : do not allow more than this many iterations
    init_tt_cores_val   : initialize cores to this number, or random if None (set np.randomseed(...)
                          before calling this function for reproducibility)

    returns:
    tt_cores, bf_vals, ys_fit
    '''

    # some diagnostics
    print_iter_res = kwargs.pop('print_iter_res')

    nnodes = len(nodes)
    bf_vals = basis_function_values(xs, nodes, kernel_f)
    tt_cores = init_tt(tt_ranks, nnodes, init_val=init_tt_cores_val)

    iter = 0
    ys_fit = predict(xs, tt_cores, kernel_f, nodes)
    r2_prev = 1 - np.linalg.norm(ys_fit - ys)/np.linalg.norm(ys)
    for iter in range(max_iter):
        tt_cores, ys_fit = solve_one_pass(
            tt_cores, bf_vals, ys, iter_n=iter, **kwargs)

        r2 = 1 - np.linalg.norm(ys_fit - ys)/np.linalg.norm(ys)
        rel_improv = r2 - r2_prev
        r2_prev = r2

        # rel_improv = np.linalg.norm(
        #     ys_fit - ys_prev_fit)/np.linalg.norm(ys_prev_fit)
        if print_iter_res:
            print(f'iter = {iter} r2 = {r2} rel_improv = {rel_improv}')
        if rel_improv <= rel_tol:
            break

        # ys_prev_fit = ys_fit.copy()

    kwargs['print_iter_res'] = print_iter_res
    return tt_cores, bf_vals, ys_fit, 1-r2_prev, iter


def predict(xs, tt_cores, kernel_f, nodes):
    '''
    Once calibrated, calculate over another set of xs
    '''
    nx = xs.shape[0]
    ndim = len(tt_cores)

    bf_vals = basis_function_values(xs, nodes, kernel_f)
    mleft = np.ones((nx, 1, 1))
    for d1 in range(ndim):
        g = np.dot(bf_vals[d1], tt_cores[d1])
        mleft = np.matmul(mleft, g)
    return np.squeeze(mleft)


def get_function(d, i, j, tt_cores, kernel_f, nodes):

    sim_range = nodes[-1]
    knl_w = tt_cores[d][i, :, j]

    def f_cheb(x):
        return cheb.chebvander(x[:, d]/sim_range, len(nodes)-1) @ knl_w

    def f(x):
        return kernel_f(x[:, d:d+1] - np.expand_dims(nodes, 0)).numpy() @ knl_w

    use_cheb = False
    if use_cheb:
        return f_cheb

    return f
