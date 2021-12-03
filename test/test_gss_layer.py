import pytest
import numpy as np
import tensorflow as tf
from matplotlib import cm
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import tensorflow.keras.backend as K
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer


from nnu import gss_layer as gssl

from nnu import nn_utils
from nnu import gss_kernels as gssk
from nnu import points_generator as pgen


def test_layer_01():
    input_shape = (5, 2)

    knots = tf.random.uniform((10, 2), minval=0, maxval=1.0, seed=314)
    pk_layer = gssl.ProdKernelLayer(knots=knots)

    x = tf.constant([[1, 2]]*5, dtype=tf.float64)

    pk_layer.build(input_shape)
    y = pk_layer(x)
    print(y.shape)
    y2 = y.numpy()
    print(y2)


def test_layer_with_one_scale_per_knot_01():
    input_shape = (5, 2)

    knots = tf.random.uniform((10, 2), minval=0, maxval=1.0, seed=314)
    pk_layer = gssl.ProdKernelLayer(
        knots=knots,
        scales=None,
        optimize_knots=False,
        optimize_scales=True,
        scales_dim=gssl.ScalesDim.OnePerKnot,
        activation=gssk.lanczos_kernel(),
    )

    x = tf.constant([[1, 2]]*5, dtype=tf.float64)

    pk_layer.build(input_shape)
    y = pk_layer(x)
    print(y.shape)
    y2 = y.numpy()
    print(y2)


def test_layer_with_one_scale_per_dim_01():
    input_shape = (5, 2)

    knots = tf.random.uniform((10, 2), minval=0, maxval=1.0, seed=314)
    pk_layer = gssl.ProdKernelLayer(
        knots=knots,
        scales=None,
        optimize_knots=False,
        optimize_scales=True,
        scales_dim=True,
        activation=gssk.lanczos_kernel(),
    )

    x = tf.constant([[1, 2]]*5, dtype=tf.float64)

    pk_layer.build(input_shape)
    y = pk_layer(x)
    print(y.shape)
    y2 = y.numpy()
    print(y2)


def test_layer_with_only_one_scale_01():
    input_shape = (5, 2)

    knots = tf.random.uniform((10, 2), minval=0, maxval=1.0, seed=314)
    pk_layer = gssl.ProdKernelLayer(
        knots=knots,
        scales=None,
        optimize_knots=False,
        optimize_scales=True,
        scales_dim=gssl.ScalesDim.OnlyOne,
        activation=gssk.lanczos_kernel(),
        average_slopes=np.array([1]*input_shape[1])
    )

    x = tf.constant([[1, 2]]*5, dtype=tf.float64)

    pk_layer.build(input_shape)
    y = pk_layer(x)
    print(y.shape)
    y2 = y.numpy()
    print(y2)


def test_model_01():

    ndim = 2
    nknots = 25
    nbatch = 10
    knots = np.random.uniform(0, 1.0, size=(nknots, ndim))
    input_shape = (nbatch, ndim)

    model = keras.Sequential()

    # our main main layer
    pk_layer = gssl.ProdKernelLayer(
        knots=knots,
        input_dim=ndim
    )
    model.add(pk_layer)

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
        use_bias=True
    ))

    x = tf.constant(np.random.uniform(0.0, 1.0, size=(nbatch, ndim)))
    model.build()
    y = model.predict(x)
    print(y.shape)

    opt = keras.optimizers.Adam()
    model.compile(loss="mean_squared_error",
                  optimizer=opt, metrics=['mse', 'mae'])

    stats = model.evaluate(x, y)
    print(stats)


@pytest.mark.parametrize('optimize_knots,optimize_scales',
                         [(False, False),
                          (True, False),
                          (False, True),
                          (True, True), ]
                         )
def test_model_02(
        optimize_knots,
        optimize_scales,
        plot_figures=False):

    seed = 2021
    nn_utils.set_seed_for_keras(seed)

    ndim = 2
    nknots = 10
    nbatch = 1000
    knots = np.random.uniform(0.0, 1.0, size=(nknots, ndim))

    scale = pgen.average_distance(knots)

    model = keras.Sequential()

    # our main fsvd layer
    model.add(gssl.ProdKernelLayer(
        input_dim=ndim,
        knots=knots,
        scales=np.ones_like(knots)*scale,
        optimize_knots=optimize_knots,
        optimize_scales=optimize_scales,
        activation=gssk.lanczos_kernel(a=2.0),
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
        use_bias=True
    ))

    x = np.random.uniform(0.0, 1.0, size=(nbatch, ndim))
    model.build()

    knl_layer_w_names = [
        weight.name for weight in model.get_layer('prodkernel').weights]
    knots_idx = [knl_layer_w_names.index(
        i) for i in knl_layer_w_names if 'knot' in i][0]
    scales_idx = [knl_layer_w_names.index(
        i) for i in knl_layer_w_names if 'scale' in i][0]

    nodes_orig = model.get_layer('prodkernel').get_weights()[
        knots_idx][:, 0].copy()

    y = (x[:, 0]-0.5)*(x[:, 1]-0.5)

    opt = keras.optimizers.Adam()
    model.compile(loss="mean_squared_error",
                  optimizer=opt, metrics=['mse', 'mae'])

    n_epochs = 1  # x100
    batch_size = nbatch//10
    learn_rate = 0.1

    opt = keras.optimizers.Adam(learning_rate=learn_rate)

    callbacks = [TqdmCallback(verbose=0)]

    start_time = timer()
    stats_before = model.evaluate(x, y)
    model.fit(x, y, epochs=100*n_epochs, batch_size=batch_size,
              verbose=0, callbacks=callbacks)
    stats_after = model.evaluate(x, y)
    fit = model.predict(x)
    end_time = timer()
    print("Time elappsed", end_time - start_time)

    nodes_new = model.get_layer('prodkernel').get_weights()[
        knots_idx][:, 0].copy()
    model.summary()
    print('config: ', optimize_knots, optimize_scales)
    print('before ', stats_before)
    print('after ', stats_after)
    print(1 - np.linalg.norm(fit[:, 0] - y)/np.linalg.norm(y))

    if plot_figures:
        plt.plot(fit[:, 0], y, '.')
        plt.title('actual vs fit')
        plt.show()

        ws = model.get_layer('final').get_weights()[0][:, 0]

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.scatter(knots[:, 0], knots[:, 1], ws,
                   cmap=cm.coolwarm, marker='.', label='nn weights')

        plt.xlabel('knots_x')
        plt.ylabel('knots_y')
        plt.legend(loc='best')
        plt.show()

        plt.plot(nodes_orig, nodes_orig, '.', label='orig')
        plt.plot(nodes_orig, nodes_new, '.', label='new')
        plt.title('New nodes vs orig')
        plt.legend(loc='best')
        plt.show()


@pytest.mark.parametrize('optimize_scales,scales_dim',
                         [(True, gssl.ScalesDim.OnePerKnot, ),
                          (True, gssl.ScalesDim.OnePerDim, ),
                          (True, gssl.ScalesDim.OnlyOne, ), ])
def test_functional_model_01(
        optimize_scales,
        scales_dim,
        plot_figures=False):
    '''
    Here we test the idea of an optimizer over scales while calculating linear coefficients in a regression
    inside tensorflow
    '''

    optimize_knots = False

    seed = 2021
    nn_utils.set_seed_for_keras(seed)

    ndim = 2
    nbatch = 1000
    inputX = np.random.uniform(0.0, 1.0, size=(nbatch, ndim))
    inputY = (inputX[:, 0]-0.5)*(inputX[:, 1]-0.5)

    nknots = 10
    knots = np.random.uniform(0.0, 1.0, size=(nknots, ndim))
    scale = pgen.average_distance(knots)

    fake_num_x = 1
    fake_dim = 1
    k_fake_input = keras.Input((fake_num_x,), name='input')

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

    # Construct  ProdKernelLayer for inputX. Here we have some trainable parameters that will later be optimized
    # most typically scales. Nodes have been pre-set
    per_coord_kernels_at_input_x = gssl.ProdKernelLayer(
        input_dim=ndim,
        knots=knots,
        #        scales=np.ones_like(knots)*scale,
        optimize_knots=optimize_knots,
        optimize_scales=optimize_scales,
        scales_dim=scales_dim,
        activation=gssk.lanczos_kernel(a=2.0),
        name='prodkernel'
    )(xvals)

    # Apply coordinatewise product to the output of ProdKernelLayer to get actual kernels for inputX
    kernels_at_input_x = keras.layers.Lambda(
        lambda x: K.prod(x, axis=2),
        name='product'
    )(per_coord_kernels_at_input_x)

    # Regress inputY on the product kernels evaluated at inputX
    regr_xy = keras.layers.Lambda(
        lambda xy: tf.linalg.lstsq(xy[0], xy[1], l2_regularizer=0.001),
        name='regr_xy'
    )([kernels_at_input_x, yvals])

    # Now predict the values of y from the regression (in the optimizer, for fixed values of scales)
    predict_y = keras.layers.Lambda(
        lambda xc: tf.matmul(xc[0], xc[1]),
        name='predict_y'
    )([kernels_at_input_x, regr_xy])

    # first build a model that predicts y so we can examine the results later
    predict_model = keras.Model(
        inputs=k_fake_input, outputs=predict_y, name="predict_y_model")
    predict_model.build(input_shape=(fake_num_x,))
    predict_model.summary()

    predict_model_output = predict_model(k_fake_input)

    # now add the residual
    resid_xy = keras.layers.Subtract(
        name='resid_xy'
    )([yvals, predict_model_output])

    # abd create a model for the residual -- this is the one to optimize against zero
    # suppose we can dispence with this and fit predict_model to inputY -- something to explore
    model = keras.Model(inputs=k_fake_input,
                        outputs=resid_xy, name="fit_model")

    model.build(input_shape=(fake_num_x,))
    model.summary()

    opt = keras.optimizers.Adam()
    model.compile(loss="mean_squared_error",
                  optimizer=opt, metrics=['mse', 'mae'])

    n_epochs = 10  # x100
    batch_size = nbatch
    learn_rate = 0.1

    opt = keras.optimizers.Adam(learning_rate=learn_rate)
    callbacks = [TqdmCallback(verbose=0)]

    fake_x = np.zeros((nbatch, fake_dim))
    fake_y = np.zeros(nbatch)

    start_time = timer()
    stats_before = model.evaluate(fake_x, fake_y, batch_size=batch_size)
    model.fit(fake_x, fake_y, epochs=n_epochs*100, batch_size=batch_size,
              verbose=0, callbacks=callbacks)
    stats_after = model.evaluate(fake_x, fake_y, batch_size=batch_size)
    fit = predict_model.predict(fake_x, batch_size=batch_size)
    end_time = timer()
    print('Time ellapsed', end_time - start_time)

    print('config: ', optimize_knots, scales_dim, optimize_scales)
    print('before ', stats_before)
    print('after ', stats_after)
    print(1 - np.linalg.norm(fit[:, 0] - inputY)/np.linalg.norm(inputY))

    if plot_figures:
        plt.plot(fit[:, 0], inputY, '.')
        plt.title('actual vs fit')
        plt.show()
