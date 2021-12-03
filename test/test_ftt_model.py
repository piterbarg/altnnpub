import pytest
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from timeit import default_timer as timer

from nnu import nn_utils
from nnu import ftt_model as fttm
from nnu import gss_kernels as gssk
from nnu import ftt_regression as fttr
from nnu import points_generator as pgen
from nnu import gss_report_generator as gssgen


def prepare_stuff_for_testing(input_f_spec, ndim=5, nx=5, nnodes=10):
    input_seed = 1917
    tt_ranks = [1] + [5]*(ndim-1) + [1]

    nn_utils.set_seed_for_keras(input_seed)
    init_val = None
    tt_cores = fttr.init_tt(tt_ranks, nnodes, init_val=init_val)

    sim_range = 4
    stretch = 1.05
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

    kernel = 'invquad'
    scale_mult = 4.0
    global_scale = 2*sim_range*stretch / nnodes
    knl_f = gssk.global_kernel_dict(global_scale * scale_mult)[kernel]
    nodes = np.linspace(-sim_range*stretch, sim_range *
                        stretch, nnodes, endpoint=True)
    bf_vals_list = fttr.basis_function_values(xs, nodes, knl_f)

    return tt_cores, bf_vals_list, tt_ranks, nodes, knl_f, xs, ys, knl_f


@pytest.mark.parametrize('input_f_spec,ndim', [('test', 3), ('laplace_0', 2), ('midrank_0', 11)])
def test_layer_01(input_f_spec, ndim):

    tt_cores, bf_vals_list, *_ = prepare_stuff_for_testing(input_f_spec, ndim)
    ndim = len(tt_cores)
    nx = bf_vals_list[0].shape[0]
    nnodes = tt_cores[0].shape[1]

    input_shape = (nx, ndim, nnodes)
    ftt_layer = fttm.FTTLayer(tt_cores)
    ftt_layer.build(input_shape)

    bf_vals = np.transpose(np.array(bf_vals_list), (1, 0, 2))
    ys = ftt_layer(bf_vals)
    print(ys.shape)
    print(ys.numpy())


@pytest.mark.parametrize('input_f_spec,ndim', [('test', 3), ('laplace_0', 2), ('midrank_0', 11)])
def test_keras_model_01(input_f_spec, ndim):

    tt_cores, bf_vals_list, *_ = prepare_stuff_for_testing(input_f_spec, ndim)
    ndim = len(tt_cores)
    nx = bf_vals_list[0].shape[0]
    nnodes = tt_cores[0].shape[1]

    input_shape = (nx, ndim, nnodes)

    keras_model = keras.Sequential()
    keras_model.add(fttm.FTTLayer(
        input_shape=input_shape[1:],
        tt_cores_init=tt_cores,
        # **kwargs,
    ))
    keras_model.build()
    keras_model.summary()
    weights = keras_model.get_weights()
    print([w.shape for w in weights])


@pytest.mark.parametrize('input_f_spec,ndim', [('test', 3)])
def test_model_01(input_f_spec, ndim, show_figures=False):

    input_seed = 1917
    nnodes = 80
    nx = 10000
    prep = prepare_stuff_for_testing(
        input_f_spec, ndim=ndim, nx=nx, nnodes=nnodes)
    tt_ranks = prep[2]
    nodes = prep[3]
    xs = prep[5]
    ys = prep[6]
    kernel_f = prep[7]

    # how many als iters for initial guess
    als_iter = 2
    als_kwargs = {'use_ridge': True,
                  'ridge_alpha': 1e-6, 'print_iter_res': True, 'use_alpha_scaling': True}

    nn_utils.set_seed_for_keras(input_seed)
    model = fttm.FTTModel(nodes, tt_ranks, kernel_f)

    stats_1 = model.evaluate(xs, ys)
    model.fit_als(xs, ys, max_iter=als_iter, **als_kwargs)

    # if als_iter == 0:
    #    model.re_init_cores_random(for_als=False)
    model.init_keras_model(name='ftt_model')
    model.summary()

    n_epochs = 1  # 10  # x100
    batch_size = nx
    learn_rate = 0.1

    opt = keras.optimizers.Adam(learning_rate=learn_rate)
    model.compile(loss="mean_squared_error",
                  optimizer=opt, metrics=['mse', 'mae'])
    callbacks = [TqdmCallback(verbose=0)]

    start_time = timer()
    stats_2 = model.evaluate(xs, ys)
    model.fit_keras(xs, ys, epochs=int(100*n_epochs), batch_size=batch_size,
                    verbose=0, callbacks=callbacks)
    stats_3 = model.evaluate(xs, ys)
    end_time = timer()
    print("Time elappsed", end_time - start_time)

    als_iter = 10
    model.fit_als(xs, ys, max_iter=als_iter, **als_kwargs)
    stats_4 = model.evaluate(xs, ys)

    print(stats_1, stats_2, stats_3, stats_4)

    fit = model.predict(xs)
    r2 = 1 - np.linalg.norm(fit[:, 0] - ys)/np.linalg.norm(ys)
    print(f'final r2 = {r2}')

    if show_figures:
        plt.plot(fit[:, 0], ys, '.')
        plt.title('actual vs fit')
        plt.show()
