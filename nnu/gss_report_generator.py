import os
import scipy
import statistics
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm
from tensorflow import keras
from functools import reduce
from numpy import linalg as LA
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from timeit import default_timer as timer
from tensorflow.python.keras.utils.layer_utils import count_params

# TF-BFGS stuff from
# https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993

import tensorflow_probability as tfp  # pip install tensorflow_probability
from tf_lbfgs import tf_keras_tfp_lbfgs

# our stuff
from nnu import points_generator as pgen
from nnu import gss_report_config as gssrconfig
from nnu import gss_model_factory, laplace_kernel, nn_utils, fit_function_factory


# a few experimental/diagnostics flags collected in one place. Do not overdo this
global_flags = {
    # Save fit results to a file for further analysis
    'save_fit_res': False,
}


def generate_inputs_and_nodes(
    ndim: int,
    nsamples: int,
    nnodes: int,
    input_f_spec: str = 'laplace_0',
    input_seed=1917,
    nsr_stretch=1.0,
    sim_range=4.0,
    points_type="random",
    nodes1d=None,
):
    '''
    Generate functions to fit from their name, as well as input xs and output ys
    '''
    nn_utils.set_seed_for_keras(input_seed)

    inputX = pgen.generate_points(
        sim_range, nsamples, ndim, points_type, seed=input_seed, nodes1d=nodes1d)[0]

    if input_f_spec == 'laplace_0':
        stds = [1.5, 1.0, 0.5][-ndim:]
        off_diag_correl = 0.0
        laplace_shift = 1
        means = np.array([[1, 1, 0], [-1, -1, 0], [0.5, -0.5, 0]])
        means = means[:, :ndim]
        cov_multipliers = [0.5, 0.3, 0.1]
        mix_weights = [0.4, 0.35, 0.35]

        kernel_type = fit_function_factory.KernelType.LpM
        covar_matr = laplace_kernel.simple_covar_matrix(stds, off_diag_correl)

        function_to_fit = fit_function_factory.generate_nd(
            kernel_type, covar_matr, shift=laplace_shift,
            means=means,
            cov_multipliers=cov_multipliers,
            mix_weights=mix_weights)

    elif input_f_spec == 'laplace_1':
        stds = [1.5, 1.0, 0.5][-ndim:]
        off_diag_correl = 0.0
        laplace_shift = 1
        means = np.array([[1, 1, 0], [-1, -1, 0], [0.5, -0.5, 0]])
        means = means[:, :ndim]
        cov_multipliers = [0.5, 0.3, 0.1]
        mix_weights = [0.4, -0.35, 0.35]

        kernel_type = fit_function_factory.KernelType.LpM
        covar_matr = laplace_kernel.simple_covar_matrix(stds, off_diag_correl)

        function_to_fit = fit_function_factory.generate_nd(
            kernel_type, covar_matr, shift=laplace_shift,
            means=means,
            cov_multipliers=cov_multipliers,
            mix_weights=mix_weights)

    elif input_f_spec.startswith('midrank'):
        # eg 'midrank_0' or 'midrank_1_w'. Here _w means we want to weight different dimensions
        # differently, at the moment using pre-canned weights. This makes the function non-symmetric
        # also supported is midrank_0_1 where '1' says how many dimensions to skip, in calculating the rank,
        #  from the start and the end. In effect the function does not depend on x[:skip_dims] and x[-skip_dims:]
        # parts of the input x

        f_spec_tokens = input_f_spec.split('_')
        rank = int(f_spec_tokens[1])
        dim_weights = None
        skip_dims = 0
        if len(f_spec_tokens) == 3 and ndim > 1:

            if f_spec_tokens[-1] == 'w':
                # uniform from 0.5 to 1.5 inclusive
                dim_weights = np.arange(ndim)/(ndim-1) + 0.5
            else:
                try:
                    skip_dims = int(f_spec_tokens[-1])
                except Exception:
                    skip_dims = 0
                    raise ValueError(
                        f'input_f_spec "{input_f_spec}" cannot be parsed, using the default')

        tau = 5e0
        function_to_fit = fit_function_factory.generate_smooth_rank_f(
            ndim, rank, tau, dim_weights, skip_dims)

    elif input_f_spec == 'doput_0':
        # down and out put with a fairly mild but relevant range of values
        spot = 100
        strike = 100
        strike_range_rel = (0.5, 1.5)
        barrier = 50
        barrier_range_rel = (0.5, 1.5)
        vol = 0.2
        vol_range_rel = (0.25, 2.5)
        expiry = 1
        expiry_range_rel = (0.5, 2)
        rate = 0.05
        rate_range_rel = (0.0, 2)

        function_to_fit = fit_function_factory.generate_down_out_put(
            ndim, sim_range, spot=spot,
            strike=strike, strike_range_rel=strike_range_rel,
            barrier=barrier, barrier_range_rel=barrier_range_rel,
            vol=vol, vol_range_rel=vol_range_rel,
            expiry=expiry, expiry_range_rel=expiry_range_rel,
            rate=rate, rate_range_rel=rate_range_rel)

    elif input_f_spec == 'doput_1':
        # down and out put with more challenging region a-la Lucic et all
        # "Backdoor Attack and Defense for Deep Regression"
        spot = 100
        strike = 100
        strike_range_rel = (0.5, 2)
        barrier = 50
        barrier_range_rel = (0.2, 2)
        vol = 0.2
        vol_range_rel = (0.05, 5)
        expiry = 1
        expiry_range_rel = (0.002, 5)
        rate = 0.05
        rate_range_rel = (0.0, 2)

        function_to_fit = fit_function_factory.generate_down_out_put(
            ndim, sim_range, spot=spot,
            strike=strike, strike_range_rel=strike_range_rel,
            barrier=barrier, barrier_range_rel=barrier_range_rel,
            vol=vol, vol_range_rel=vol_range_rel,
            expiry=expiry, expiry_range_rel=expiry_range_rel,
            rate=rate, rate_range_rel=rate_range_rel)

    else:
        raise ValueError(f'unknown input function spec:{input_f_spec}')

    inputY = function_to_fit(inputX)

    nodes_type = "random"
    nodes = pgen.generate_points(
        sim_range*nsr_stretch, nnodes, ndim, nodes_type, seed=input_seed, plot=0, min_dist=0.3)[0]
    global_scale = pgen.average_distance(nodes)

    return inputX, inputY, nodes, global_scale, function_to_fit


def generate_results(
    model_name: str,
    ndim: int,
    nsamples: int,
    nnodes: int,
    epochs=1,
    kernel='lanczos',
    input_f_spec='laplace_0',
    seed_for_keras=2021,
    input_seed=1917,
    test_res_seed=314,
    test_res_runs=1,
    show_figures=False,
    l2_multiplier=1,
    tf_dtype=tf.float32,
    nsr_stretch=1.0,
    batch_label='',
    generate_more_nodes=True,
    sim_range=4.0,

):
    '''
    The main dispatch function that creates the right model and fits it to given inputs
    '''
    inputX, inputY, nodes, global_scale, *_ = generate_inputs_and_nodes(
        ndim, nsamples, nnodes, input_f_spec, input_seed, nsr_stretch, sim_range)

    model_specs = gssrconfig.get_model_specs(kernel)[model_name]
    opt_specs = gssrconfig.get_optimizer_specs()[model_name]
    np_dtype = np.float32 if tf_dtype == tf.float32 else np.float64

    # breaking encapsulation here a bit, but this is needed for '1d' and eventual reporting
    l2_regularizer = model_specs['l2_regularizer']
    l2_regularizer *= l2_multiplier

    init_scales = None

    model = gss_model_factory.generate_model(
        ndim=ndim,
        global_scale=global_scale,
        nodes=nodes,
        inputX=inputX,
        inputY=inputY,
        scales=init_scales,
        seed_for_keras=seed_for_keras,
        l2_multiplier=l2_multiplier,
        sim_range=sim_range,
        **model_specs)

    model_epochs = epochs * opt_specs['epoch_mult']
    learn_rate = opt_specs['learn_rate']

    if (opt_specs['is_regr']):

        fake_dim = 1
        output_size = 1 if model_specs['apply_final_aggr'] else nsamples
        x_to_use = np.zeros((output_size, fake_dim), dtype=np_dtype)
        y_to_use = np.zeros(output_size, dtype=np_dtype)

    else:
        x_to_use = inputX
        y_to_use = inputY

    nn_utils.set_seed_for_keras(seed_for_keras)
    opt_type = opt_specs['type']
    if opt_type == 'adam':
        batch_size = nsamples

        opt = keras.optimizers.Adam(learning_rate=learn_rate)
        model.compile(loss="mean_squared_error",
                      optimizer=opt, metrics=['mse', 'mae'])

        callbacks = [TqdmCallback(verbose=0)]

        start_time = timer()
        model.fit(x_to_use, y_to_use, epochs=int(100*model_epochs), batch_size=batch_size,
                  verbose=0, callbacks=callbacks)
        end_time = timer()

    elif opt_type == 'bfgs':

        adam_epochs = int(100*model_epochs)
        adam_lr = 0.1
        adam_batch_size = nsamples

        print_params = False

        if model_name == 'lodim_regr_bfgs' or model_name == 'onedim_regr_bfgs':
            print_params = True

        print('Adam is preparing a good starting point for BFGS ...')

        # start with Adam to get going for a few iterations
        start_time = timer()
        opt = keras.optimizers.Adam(learning_rate=adam_lr)
        model.compile(loss="mean_squared_error",
                      optimizer=opt, metrics=['mse', 'mae'])

        callbacks = [TqdmCallback(verbose=0)]

        model.fit(x_to_use, y_to_use, epochs=adam_epochs, batch_size=adam_batch_size,
                  verbose=0, callbacks=callbacks)

        print('...Now BGFS is ready to take over')

        # get the wrapper up and ready
        loss_fun = tf.keras.losses.MeanSquaredError()
        func = tf_keras_tfp_lbfgs.function_factory(
            model, loss_fun, x_to_use, y_to_use, print_params=print_params)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(
            func.idx, model.trainable_variables)

        # train the model with L-BFGS solver
        try:
            results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function=func,
                initial_position=init_params,
                tolerance=tf.constant(1e-6, dtype=tf_dtype),
                x_tolerance=tf.constant(1e-6, dtype=tf_dtype),
                f_relative_tolerance=tf.constant(1e-6, dtype=tf_dtype),
                parallel_iterations=4,  # ?
                max_iterations=int(50*model_epochs)
            )
            # after training, the final optimized parameters are still in results.position
            # so we have to manually put them back to the model
            # presumably this sets the "model" with the new params as well
            func.assign_new_model_parameters(results.position)

        except Exception as e:
            print(f'BFGS failed with exception: {e}')

        end_time = timer()

    elif opt_type == 'bfgs_1d':
        start_time = timer()

        # Load a 'test' model to calculate gradients. The test model
        # will have its outer weights (linear coefficients) calculated by regression
        # so this is the "base" approximation from the paper
        grad_model = gss_model_factory.generate_model_for_testing_2(
            model,
            ndim=ndim,
            global_scale=global_scale,
            nodes=nodes,
            **model_specs,
            tf_dtype=tf_dtype,
            nsr_stretch=nsr_stretch,
            generate_more_nodes=False, sim_range=sim_range)

        # Following https://stackoverflow.com/a/59591167/14551426
        x_tensor = tf.convert_to_tensor(inputX, dtype=tf_dtype)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            output = grad_model(x_tensor)
            gradients = t.gradient(output, x_tensor).numpy()

        average_slopes = np.linalg.norm(gradients, axis=0)
        average_slopes /= np.amax(average_slopes)
        print("directional risk magnitudes: ", average_slopes)
        print_params = True

        # Redefine the model now that we know average_slopes
        model = gss_model_factory.generate_model(
            ndim=ndim,
            global_scale=global_scale,
            nodes=nodes,
            inputX=inputX,
            inputY=inputY,
            seed_for_keras=seed_for_keras,
            l2_multiplier=l2_multiplier,
            average_slopes=average_slopes,
            sim_range=sim_range,
            **model_specs)

        # get the wrapper up and ready
        loss_fun = tf.keras.losses.MeanSquaredError()
        func = tf_keras_tfp_lbfgs.function_factory(
            model, loss_fun, x_to_use, y_to_use, print_params=print_params)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(
            func.idx, model.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            tolerance=1e-6,
            x_tolerance=1e-6,
            f_relative_tolerance=1e-6,
            parallel_iterations=4,  # ?
            max_iterations=int(25*model_epochs)
        )
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        # presumably this sets the "model" with the new params as well
        func.assign_new_model_parameters(results.position)

        end_time = timer()

    elif opt_type == '1d':

        start_time = timer()
        # Load a 'test' model to calculate gradients. The test model
        # will have its outer weights (linear coefficients) calculated by regression
        # so this is the "base" approximation from the paper
        grad_model = gss_model_factory.generate_model_for_testing_2(
            model,
            ndim=ndim,
            global_scale=global_scale,
            nodes=nodes,
            **model_specs,
            tf_dtype=tf_dtype,
            nsr_stretch=nsr_stretch,
            generate_more_nodes=False, sim_range=sim_range)

        # Following https://stackoverflow.com/a/59591167/14551426
        x_tensor = tf.convert_to_tensor(inputX, dtype=tf_dtype)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            output = grad_model(x_tensor)
            gradients = t.gradient(output, x_tensor).numpy()

        average_slopes = np.linalg.norm(gradients, axis=0)
        average_slopes /= np.amax(average_slopes)
        print("directional risk magnitudes: ", average_slopes)
        print_params = True

        # Redefine the model now that we know average_slopes
        model = gss_model_factory.generate_model(
            ndim=ndim,
            global_scale=global_scale,
            nodes=nodes,
            inputX=inputX,
            inputY=inputY,
            seed_for_keras=seed_for_keras,
            l2_multiplier=l2_multiplier,
            average_slopes=average_slopes,
            sim_range=sim_range,
            **model_specs)

        # Get access to the inner weight corresponding to the common scale (known to be a scalar in this particular case)
        #  that drives all scales so we can set it directly and optimize in a 1d optimizer
        predict_model = model.get_layer('predict_y_model')
        prod_kernel = predict_model.get_layer('prodkernel')
        orig_weights = prod_kernel.get_weights()

        def obj_f_1d(w):
            '''
            1D objective function for scipy.optimize.minimize_scalar
            '''
            # set the right weight to betas, reuse the others (nodes -- not trainable here)
            prod_kernel.set_weights([np.array([[w]]), orig_weights[1]])

            fit = predict_model.predict(x_to_use, batch_size=nsamples)
            mse = np.linalg.norm(fit[:, 0] - inputY)/np.linalg.norm(inputY)
            return mse

        res = scipy.optimize.minimize_scalar(
            obj_f_1d, bounds=(0.25, 1.25), method='bounded', options={'xatol': 1e-1})

        # This sets the scale to the final solution. Prints it for good measure
        achieved_mse = obj_f_1d(res.x)
        print(
            f'1d optimizer found solution {res.x} and achieved mse = {achieved_mse}')

        end_time = timer()

    elif opt_type == 'als':
        start_time = timer()
        nn_utils.set_seed_for_keras(seed_for_keras)
        model.re_init_cores_random(for_als=True)

        als_iter = model_epochs
        ridge_alpha = 1e-6*(len(model.nodes)/100.0)**2
        # ridge_alpha = 1e-3*(len(model.nodes)/100.0)**2
        als_kwargs = {'use_ridge': True,
                      'ridge_alpha': ridge_alpha, 'print_iter_res': True, 'use_alpha_scaling': True,
                      'learning_rate': learn_rate, }

        model.fit_als(x_to_use, y_to_use, max_iter=als_iter, **als_kwargs)

        end_time = timer()

    else:
        raise ValueError(f'unknown optimizer type: {opt_type}')

    fitting_time = end_time - start_time

    predict_model = None
    if (opt_specs['is_regr']):
        predict_model = model.get_layer('predict_y_model')
    else:
        predict_model = model

    fit = predict_model.predict(x_to_use, batch_size=nsamples)
    learn_mse = np.linalg.norm(fit[:, 0] - inputY)/np.linalg.norm(inputY)
    learn_mae = np.linalg.norm(
        fit[:, 0] - inputY, ord=1)/np.linalg.norm(inputY)/np.sqrt(nsamples)  # note we divide by L2 norm on purpose

    save_fit_res = global_flags['save_fit_res']
    if save_fit_res:
        df_fr = pd.DataFrame(
            columns=[f'x[{n}]' for n in range(inputX.shape[1])], data=inputX)
        df_fr['act'] = inputY
        df_fr['fit'] = fit[:, 0]
        df_fr['err'] = fit[:, 0] - inputY
        nn_utils.df_save_to_results(
            df_fr, file_name=f'fit_res_{model_name}_{input_f_spec}_{ndim}.csv',)
        if show_figures:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            ax.scatter(inputX[:, 0], inputX[:, 1], inputX[:, 2],
                       c=fit[:, 0] - inputY,
                       cmap=cm.coolwarm, marker='.', label='fit - act')

            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.set_zlabel('x[2]')
            plt.legend(loc='best')
            plt.show()

    n_params = count_params(model.trainable_weights)

    test_model = gss_model_factory.generate_model_for_testing_2(
        model, ndim, global_scale, nodes, **model_specs,
        tf_dtype=tf_dtype, nsr_stretch=nsr_stretch,
        generate_more_nodes=generate_more_nodes, sim_range=sim_range, epochs=epochs)

    test_errors = []
    test_maes = []
    testing_time = 0.0
    for s in range(test_res_runs):
        testX, testY, *_ = generate_inputs_and_nodes(
            ndim, nsamples, nnodes, input_f_spec, test_res_seed+s, nsr_stretch, sim_range)

        start_time = timer()
        test_fit = test_model.predict(testX)
        end_time = timer()
        testing_time += end_time - start_time

        mse = np.linalg.norm(test_fit[:, 0] - testY)/np.linalg.norm(testY)
        mae = np.linalg.norm(
            test_fit[:, 0] - testY, ord=1)/np.linalg.norm(testY)/np.sqrt(nsamples)  # note we divide by L2 norm on purpose
        test_errors.append(mse)
        test_maes.append(mae)

    test_mse = statistics.mean(test_errors)
    test_mae = statistics.mean(test_maes)
    test_mse_stdev = 0.0
    if len(test_errors) > 1:
        test_mse_stdev = statistics.stdev(test_errors)

    mc_error = 1-np.linalg.norm(inputY)/np.linalg.norm(testY)

    if show_figures:
        plt.plot(fit[:, 0], inputY, '.', label='actual vs fit')
        plt.title(model_name)
        plt.show()

    res = {
        'learn_mse': learn_mse,
        'learn_mae': learn_mae,
        'fitting_time': fitting_time,
        'n_params': n_params,
        'n_epochs': model_epochs,
        'mc_error': mc_error,
        'test_mse': test_mse,
        'test_mse_stdev': test_mse_stdev,
        'test_mae': test_mae,
        'input_f_spec': input_f_spec,
        'ndim': ndim,
        'l2_regularizer': l2_regularizer,
        'nsr_stretch': nsr_stretch,
        'kernel': kernel,
        'nsamples': nsamples,
        'model': model_name,
        'kernel': kernel,
        'seed_keras': seed_for_keras,
        'seed_input': input_seed,
        'batch_label': batch_label,
        'testing_time': testing_time,

    }

    return res, fit,


def generate_results_all_models(
        run_name: str = '',
        run_spec: dict = None,
        batch_spec: dict = None,
):
    if run_spec is None:
        run_spec = gssrconfig.get_run_specs(
            nnodes=10, epochs=1, get_user_defined=True)['default']

    if batch_spec is None:
        batch_spec = gssrconfig.get_standard_batch_specs()

    model_names = batch_spec['model_names']
    all_model_names = gssrconfig.get_model_specs().keys()
    if model_names is None:
        model_names = all_model_names

    # do not need these anymore for this run but will need for other runs, so make a copy
    batch_spec_copy = batch_spec.copy()
    batch_spec_copy.pop('model_names')
    batch_spec_copy.pop('run_names')

    dict_out = {}
    for model_name in model_names:
        if model_name not in all_model_names:
            print(f'Model name {model_name} not recognized')
            continue

        # double the multiplier until it works. Try at most 11 times (x1...x1024)
        it_worked = False
        for l2_power_mult in range(11):
            l2_multiplier = 2**l2_power_mult
            l2_regularizer = gssrconfig.get_model_specs()[
                model_name]['l2_regularizer']
            l2_regularizer *= l2_multiplier
            batch_spec_copy['l2_multiplier'] = l2_multiplier
            try:
                res, _ = generate_results(
                    model_name=model_name,
                    **run_spec,
                    **batch_spec_copy,
                )

                res['run_id'] = run_name
                dict_out[model_name] = res

                it_worked = True
                break
            except Exception as e:
                it_worked = False
                print(
                    f'\nl2_regularizer = {l2_regularizer} failed with exception {e}')

        if not it_worked:

            dict_out[model_name] = {
                'fitting_time': 'fail',
                'test_mse': 'fail',
                'l2_regularizer': l2_regularizer,
                'run_id': run_name,
            }

        print('model_name:', model_name)
        print('run specs:', run_spec)
        print(dict_out[model_name])

    df_out = pd.DataFrame.from_dict(dict_out, orient='index')
    print(df_out)
    return df_out, dict_out


def generate_results_all_runs(
    short_run=False,
    file_name='gss_latest_results.csv',
    batch_spec: dict = None,
):
    if batch_spec is None:
        batch_spec = gssrconfig.get_standard_batch_specs()
    tf_dtype = batch_spec['tf_dtype']
    run_names = batch_spec['run_names']
    batch_label = batch_spec['batch_label']

    keras_dtype = 'float32' if tf_dtype == tf.float32 else 'float64'
    tf.keras.backend.set_floatx(keras_dtype)
    # np_dtype = np.float32 if keras_dtype == 'float32' else np.float64

    all_run_specs = gssrconfig.get_run_specs(short_run=short_run)

    if run_names is None:
        run_specs = all_run_specs
    else:
        run_specs = {k: all_run_specs[k]
                     for k in run_names if k in all_run_specs.keys()}

    res_list = []
    for key, run_spec in run_specs.items():

        print('#########################################################')
        print(f'key={key}')

        _, model_dict = generate_results_all_models(
            run_name=key, run_spec=run_spec,  batch_spec=batch_spec)

        for _, res_dict in model_dict.items():
            res_list.append(res_dict)

    res_df = pd.DataFrame.from_records(res_list)
    nn_utils.df_save_to_results(
        df=res_df, file_name=file_name, suffix=batch_label, add_date=True)

    return res_df, res_list
