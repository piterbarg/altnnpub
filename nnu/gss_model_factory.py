import math
import numpy as np
import tensorflow as tf
from scipy import optimize
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn.linear_model import Ridge


# our stuff
from nnu import gss_kernels as gssk
from nnu import points_generator as pgen
from nnu import nn_utils, gss_layer, gss_report_config, ftt_model


def generate_model(
    ndim: int,
    global_scale: float,
    nodes: np.ndarray,
    inputX=None,
    inputY=None,
    scales=None,
    model_type=gss_report_config.ModelType.SSL,
    use_outer_regression=True,
    optimize_knots=False,
    optimize_scales=True,
    scales_dim=gss_layer.ScalesDim.OnePerKnot,
    apply_final_aggr=False,
    kernel='lanczos',
    average_slopes=None,
    seed_for_keras=2021,
    l2_regularizer=1e-4,
    l2_multiplier=1,
    tf_dtype=tf.float32,
    sim_range=4,
):
    l2_regularizer *= l2_multiplier

    # if generate_equiv_relu_nn:
    if model_type == gss_report_config.ModelType.ReLU:
        return generate_relu_nn_model(ndim, nodes, use_outer_regression, optimize_knots, optimize_scales, scales_dim, seed_for_keras)

    if model_type == gss_report_config.ModelType.FTT:
        return generate_ftt_model(
            ndim, nodes, scales_dim, kernel, seed_for_keras, sim_range=sim_range)

    # if generate_bespoke_model:
    # if model_type == gss_report_config.ModelType.Custom1D:
    #    b = np.array([.5]*ndim)  # good guess
    #    freq_bounds = b/global_scale
    #    freq_bounds_vec = np.array([freq_bounds]*len(nodes))
    #
    #    y_nodes = np.zeros(len(nodes))
    #    stoch_sample_model = stoch_sampling.StochSampling(
    #        nodes,
    #        freq_bounds_vec,
    #        y_nodes,
    #        # kernel going to stoch_sampling is *not* scaled
    #        global_kernel_dict(1.0, tf_dtype=tf_dtype)[kernel],
    #        l2_regularizer)
    #
    #    # linear learn
    #    stoch_sample_model.regressTF(inputX, inputY)
    #    return stoch_sample_model
    #
    if not use_outer_regression:
        return generate_non_regr_model(ndim, global_scale, nodes, optimize_knots, optimize_scales, scales_dim, kernel, average_slopes=average_slopes, seed_for_keras=seed_for_keras)

    # the main event -- regression based model
    if inputX is None:
        raise ValueError('Need inputX for the regression model')
    if inputY is None:
        raise ValueError('Need inputY for the regression model')

    fake_dim = 1
    k_fake_input = keras.Input((fake_dim,), name='input')

    # "sideload" inputX -- a node that always returns inputX
    xvals = keras.layers.Lambda(
        lambda _: tf.constant(inputX),
        input_dim=fake_dim,
        name='xpts'
    )(k_fake_input)

    # "sideload" inputY -- a node that always returns inputY
    yvals = keras.layers.Lambda(
        lambda _: tf.expand_dims(tf.constant(inputY), -1),
        input_dim=fake_dim,
        name='ypts'
    )(k_fake_input)

    activation = gssk.global_kernel_dict(
        global_scale, tf_dtype=tf_dtype)[kernel]

    # Construct  ProdKernelLayer for inputX. Here we have some trainable parameters that will later be optimized
    # most typically scales. Nodes have been pre-set
    per_coord_kernels_at_input_x = gss_layer.ProdKernelLayer(
        input_dim=ndim,
        knots=nodes,
        scales=scales,
        optimize_knots=optimize_knots,
        optimize_scales=optimize_scales,
        scales_dim=scales_dim,
        activation=activation,
        average_slopes=average_slopes,
        name='prodkernel'
    )(xvals)

    # Apply coordinatewise product to the output of ProdKernelLayer to get actual kernels for inputX
    kernels_at_input_x = keras.layers.Lambda(
        lambda x: K.prod(x, axis=2),
        name='product'
    )(per_coord_kernels_at_input_x)

    def my_lstsq(A, y, l2_regularizer=1e-8):
        A_T = tf.transpose(A)
        lhs = tf.linalg.matmul(A_T, A) + l2_regularizer * \
            tf.linalg.eye(A.shape[1])
        rhs = tf.linalg.matmul(A_T, y)
        return tf.linalg.solve(lhs, rhs)

    # Regress inputY on the product kernels evaluated at inputX
    regr_xy = keras.layers.Lambda(
        lambda xy: my_lstsq(
            xy[0], xy[1], l2_regularizer=l2_regularizer),
        # lambda xy: tf.linalg.lstsq(
        #     xy[0], xy[1], l2_regularizer=l2_regularizer),
        name='regr_xy'
    )([kernels_at_input_x, yvals])

    # build a regression model so we can extract the outer coefs
    regr_model = keras.Model(
        inputs=k_fake_input, outputs=regr_xy, name="regr_xy_model")
    regr_model.build(input_shape=(fake_dim,))
    regr_model_output = regr_model(k_fake_input)

    # Now predict the values of y from the regression (in the optimizer, for fixed values of scales)
    predict_y = keras.layers.Lambda(
        lambda xc: tf.matmul(xc[0], xc[1]),
        name='predict_y'
    )([kernels_at_input_x, regr_model_output])

    # first build a model that predicts y so we can examine the results later
    predict_model = keras.Model(
        inputs=k_fake_input, outputs=predict_y, name="predict_y_model")

    nn_utils.set_seed_for_keras(seed_for_keras)
    predict_model.build(input_shape=(fake_dim,))

    predict_model_output = predict_model(k_fake_input)

    # now add the residual
    resid_xy = keras.layers.Subtract(
        name='resid_xy'
    )([yvals, predict_model_output])

    # optionally sum up the squares inside the model
    if apply_final_aggr:
        resid_xy = keras.layers.Lambda(
            lambda z: tf.reduce_sum(tf.square(z)),
            name='sumsq'
        )(resid_xy)

    # abd create a model for the residual -- this is the one to optimize against zero
    # suppose we can dispence with this and fit predict_model to inputY -- something to explore
    model = keras.Model(inputs=k_fake_input,
                        outputs=resid_xy, name="fit_model")

    model.build(input_shape=(fake_dim,))

    return model


def generate_relu_nn_model(
    ndim: int,
    nodes: np.ndarray,
    use_outer_regression=True,
    optimize_knots=False,
    optimize_scales=True,
    scales_dim=gss_layer.ScalesDim.OnePerKnot,
    seed_for_keras=2021
):

    # figure out total weights
    n_nodes = nodes.shape[0]
    n_total_w = n_nodes*(1-use_outer_regression) + n_nodes*ndim*optimize_knots
    if scales_dim == gss_layer.ScalesDim.OnePerKnot:
        n_total_w += n_nodes*ndim*optimize_scales
    elif scales_dim == gss_layer.ScalesDim.OnePerDim:
        n_total_w += ndim*optimize_scales
    else:
        n_total_w += optimize_scales

    # Heuristics to go from  the total (approx) number of paramsto how many layers/nodes per layer we want
    # Based on "Error bounds for approximations with deep ReLU networks" by Yarotsky, Thm 1 to shape the DNN. Smoothness (Sobolev) = n
    # Then n_layers ~ ln(1/eps)+1, n_total_w ~ eps^{-d/n}*n_layers = eps^{-d}*(ln(1/eps)+1)

    # let's make sure we have some reasonable number to begin with
    n_total_w = max(n_total_w, 10)
    yar_c = 1
    sob_n = 2.0

    def nl_f(eps): return yar_c*(math.log(1/eps)+1)
    def ntw_f(eps): return nl_f(eps)*(1/eps)**(ndim/sob_n) - n_total_w

    impl_eps = optimize.brentq(ntw_f, 1e-8, 1e8)
    n_layers = int(math.floor(nl_f(impl_eps)))
    n_w_per_layer = int(math.ceil(math.sqrt(n_total_w/n_layers)))+1

    # now build the model
    model = keras.Sequential()
    activation = 'relu'

    for n in range(n_layers):

        if n == 0:
            model.add(keras.layers.Dense(
                input_dim=ndim,
                units=n_w_per_layer,
                activation=activation,
                name=f'dl_{n}'
            ))
        else:
            model.add(keras.layers.Dense(
                units=n_w_per_layer,
                activation=activation,
                name=f'dl_{n}'
            ))

    # final aggregation
    model.add(keras.layers.Dense(
        1,
        activation='linear',
        name='final',
    ))

    nn_utils.set_seed_for_keras(seed_for_keras)

    model.build()
    return model


def generate_ftt_model(
    ndim: int,
    nodes: np.ndarray,
    scales_dim=gss_layer.ScalesDim.OnePerKnot,
    kernel='lanczos',
    seed_for_keras=2021,
    sim_range=4.0,
):
    nnodes_input = len(nodes)
    nnodes = nnodes_input//10

    stretch = 1.05  # hardcode this one here
    nodes_per_dim = np.linspace(-sim_range * stretch,
                                sim_range * stretch, nnodes, endpoint=True)

    scale_mult = 10.0  # hardcoded, higher than for other models but seems to work well
    global_scale = 2*sim_range*stretch / nnodes
    kernel_f = gssk.global_kernel_dict(global_scale * scale_mult)[kernel]

    # coopt ScalesDim as our switch
    flat_rank_lookup = {
        gss_layer.ScalesDim.OnlyOne: 3,
        gss_layer.ScalesDim.OnePerDim: 4,
        gss_layer.ScalesDim.OnePerKnot: 5,

    }
    flat_rank = flat_rank_lookup[scales_dim]
    tt_ranks = [1] + [flat_rank]*(ndim-1) + [1]

    nn_utils.set_seed_for_keras(seed_for_keras)
    model = ftt_model.FTTModel(nodes_per_dim, tt_ranks, kernel_f)
    # let's do that by default because will be easier to switch for als specifically
    nn_utils.set_seed_for_keras(seed_for_keras)
    model.re_init_cores_random(for_als=False)
    model.init_keras_model(name='ftt_model')
    return model


def generate_non_regr_model(
    ndim: int,
    global_scale: float,
    nodes: np.ndarray,
    optimize_knots=False,
    optimize_scales=True,
    scales_dim=gss_layer.ScalesDim.OnePerKnot,
    kernel='lanczos',
    average_slopes=None,
    seed_for_keras=2021,
):

    activation = gssk.global_kernel_dict(global_scale)[kernel]

    model = keras.Sequential()

    # our main layer
    model.add(gss_layer.ProdKernelLayer(
        input_dim=ndim,
        knots=nodes,
        optimize_knots=optimize_knots,
        optimize_scales=optimize_scales,
        scales_dim=scales_dim,
        activation=activation,
        average_slopes=average_slopes,
        name='prodkernel'
    ))

    # coordniate-wise product
    model.add(keras.layers.Lambda(
        lambda x: K.prod(x, axis=2),
        name='product'
    ))

    # final aggregation
    model.add(keras.layers.Dense(
        1,
        activation='linear',
        name='final',
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.1, maxval=0.1, seed=None)
    ))

    nn_utils.set_seed_for_keras(seed_for_keras)

    model.build()
    return model


def generate_model_for_testing(
    fit_model,
    ndim: int,
    global_scale: float,
    nodes: np.ndarray,
    model_type=gss_report_config.ModelType.SSL,
    use_outer_regression=True,
    optimize_knots=False,
    optimize_scales=True,
    scales_dim=gss_layer.ScalesDim.OnePerKnot,
    apply_final_aggr=False,
    kernel='lanczos',
    average_slopes=None,
    l2_regularizer=1e-2,
):
    '''
    Create a version of the model that can be used to estimate test errors on different inputX, inputY
    This is only relevant for a regression-based model where we need to jump a few hoops. For others
    we just return the fit_model
    '''

    if model_type in (gss_report_config.ModelType.ReLU,
                      gss_report_config.ModelType.FTT) \
            or not use_outer_regression:
        return fit_model

    output_shape = fit_model.layers[-1].output_shape
    if output_shape == ():
        output_size = 1
    else:
        output_size = output_shape[0]

    fake_dim = 1
    x_to_use = np.zeros((output_size, fake_dim))
    # y_to_use = np.zeros(output_size)

    predict_model = fit_model.get_layer(
        'predict_y_model')
    regr_model = predict_model.get_layer('regr_xy_model')
    inner_weights = regr_model.get_layer('prodkernel').get_weights()
    outer_weights = regr_model.predict(x_to_use, batch_size=output_size)

    # get the activation function
    activation = gssk.global_kernel_dict(global_scale)[kernel]

    # start creating a test model
    test_model = keras.Sequential()

    test_model.add(gss_layer.ProdKernelLayer(
        input_dim=ndim,
        knots=nodes,
        optimize_knots=optimize_knots,
        optimize_scales=optimize_scales,
        scales_dim=scales_dim,
        activation=activation,
        average_slopes=average_slopes,
        name='prodkernel'
    ))

    # coordniate-wise product
    test_model.add(keras.layers.Lambda(
        lambda x: K.prod(x, axis=2),
        name='product'
    ))

    # final aggregation
    test_model.add(keras.layers.Dense(
        1,
        activation='linear',
        name='final',
        use_bias=False,
    ))

    test_model.build()

    # set the weights to that from the fit_model
    test_model.get_layer('prodkernel').set_weights(inner_weights)
    test_model.get_layer('final').set_weights([outer_weights])

    # Apply the 'final accurate regression' idea

    inputX = regr_model.get_layer('xpts')(x_to_use).numpy()
    inputY = regr_model.get_layer('ypts')(x_to_use).numpy()
    regr_knl = test_model.get_layer('product')(
        test_model.get_layer('prodkernel')(inputX)).numpy()
    lin_regr = Ridge(alpha=1e-8, fit_intercept=False)
    regr_res = lin_regr.fit(regr_knl, inputY)
    new_outer_weights = regr_res.coef_.T

    print('orig', np.linalg.norm(regr_knl@outer_weights - inputY))
    print('fine', np.linalg.norm(regr_knl@new_outer_weights - inputY))

    test_model.get_layer('final').set_weights([new_outer_weights])

    return test_model


def generate_model_for_testing_2(
    fit_model,
    ndim: int,
    global_scale: float,
    nodes: np.ndarray,
    model_type=gss_report_config.ModelType.SSL,
    use_outer_regression=True,
    optimize_knots=False,
    optimize_scales=True,
    scales_dim=gss_layer.ScalesDim.OnePerKnot,
    apply_final_aggr=False,
    kernel='lanczos',
    average_slopes=None,
    l2_regularizer=1e-2,
    tf_dtype=tf.float32,
    nsr_stretch=1.0,
    generate_more_nodes=True,
    sim_range=4.0,
    epochs=1.0,
):
    '''
    Create a version of the model that can be used to estimate test errors on different inputX, inputY
    This is only relevant for a regression-based model where we need to jump a few hoops. For others
    we just return the fit_model

    Here we implement the  idea of increasing the number of nodes in test_model
    '''

    # if generate_equiv_relu_nn or generate_equiv_fsvd or generate_bespoke_model or not use_outer_regression:
    if model_type in (gss_report_config.ModelType.ReLU,
                      gss_report_config.ModelType.FTT) \
            or not use_outer_regression:
        return fit_model

    output_shape = fit_model.layers[-1].output_shape
    if output_shape == ():
        output_size = 1
    else:
        output_size = output_shape[0]

    fake_dim = 1
    x_to_use = np.zeros((output_size, fake_dim))

    predict_model = fit_model.get_layer(
        'predict_y_model')
    regr_model = predict_model.get_layer('regr_xy_model')
    prod_kernel = regr_model.get_layer('prodkernel')

    if prod_kernel.scales_dim == gss_layer.ScalesDim.OnePerKnot:
        # use the original version
        return generate_model_for_testing(
            fit_model=fit_model,
            ndim=ndim,
            global_scale=global_scale,
            nodes=nodes,
            model_type=model_type,
            use_outer_regression=use_outer_regression,
            optimize_knots=optimize_knots,
            optimize_scales=optimize_scales,
            scales_dim=scales_dim,
            apply_final_aggr=apply_final_aggr,
            kernel=kernel,
            average_slopes=average_slopes,
            l2_regularizer=l2_regularizer,
        )

    inputX = regr_model.get_layer('xpts')(x_to_use).numpy()
    inputY = regr_model.get_layer('ypts')(x_to_use).numpy()

    # start creating a test model
    test_model = keras.Sequential()

    if generate_more_nodes:
        # this needs cleaning up -- params should come from somewhere not hardcoded
        new_sim_range = sim_range*nsr_stretch
        input_seed = 1917
        n_new_nodes = max(inputX.shape[0]//4, nodes.shape[0])

        # for epochs>2 increase more_nnodes
        if epochs > 2:
            n_new_nodes *= epochs-1

        new_nodes = pgen.generate_points(
            new_sim_range, n_new_nodes, ndim, "random", seed=input_seed, plot=0, min_dist=0.3)[0]
        new_global_scale = pgen.average_distance(new_nodes)
        new_activation = gssk.global_kernel_dict(
            new_global_scale, tf_dtype=tf_dtype)[kernel]

    else:
        new_nodes = nodes
        new_activation = prod_kernel.activation

    test_model.add(gss_layer.ProdKernelLayer(
        input_dim=ndim,
        knots=new_nodes,
        scales=prod_kernel.scales,
        optimize_knots=prod_kernel.optimize_knots,
        optimize_scales=prod_kernel.optimize_scales,
        scales_dim=prod_kernel.scales_dim,
        activation=new_activation,
        average_slopes=prod_kernel.average_slopes,
        name='prodkernel'
    ))

    # coordniate-wise product
    test_model.add(keras.layers.Lambda(
        lambda x: K.prod(x, axis=2),
        name='product'
    ))

    # final aggregation
    test_model.add(keras.layers.Dense(
        1,
        activation='linear',
        name='final',
        use_bias=False,
    ))

    test_model.build()

    regr_knl = test_model.get_layer('product')(
        test_model.get_layer('prodkernel')(inputX)).numpy()
    lin_regr = Ridge(alpha=1e-8, fit_intercept=False)
    regr_res = lin_regr.fit(regr_knl, inputY)
    new_outer_weights = regr_res.coef_.T

    test_model.get_layer('final').set_weights([new_outer_weights])

    return test_model
