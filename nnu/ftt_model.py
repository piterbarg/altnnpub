import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from timeit import default_timer as timer

from nnu import nn_utils
from nnu import ftt_regression as fttr
from nnu import points_generator as pgen
from nnu import gss_model_factory as gssmf
from nnu import gss_report_generator as gssgen


class FTTLayer(keras.layers.Layer):

    def __init__(self,
                 tt_cores_init,
                 **kwargs):

        super(FTTLayer, self).__init__(**kwargs)

        # a list of cores as in tff_regression.py
        self.tt_cores_init = tt_cores_init
        self.ndim = len(tt_cores_init)
        self.nnodes = tt_cores_init[0].shape[1]

    def build(self, input_shape):
        '''
        input_shape is n_batches x ndim x nnodes
        '''
        if input_shape[1] != self.ndim:
            raise ValueError(
                f'Incompatible dimension for input_shape[1], got {input_shape[1]}, expecting {self.ndim}')
        if input_shape[2] != self.nnodes:
            raise ValueError(
                f'Incompatible dimension for input_shape[2], got {input_shape[2]}, expecting {self.nnodes}')

        self.tt_cores = []
        for n, tt_core_init in enumerate(self.tt_cores_init):
            tt_core = self.add_weight(
                f'tt_core[{n}]',
                shape=tt_core_init.shape,
                dtype=self.dtype,
                trainable=True,
            )
            tt_core.assign(tt_core_init)
            self.tt_cores.append(tt_core)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0])

    def call(self, bf_vals, mask=None):

        # nx = bf_vals.shape[0]

        mleft = tf.ones((tf.shape(bf_vals)[0], 1, 1), dtype=self.dtype)
        for d1 in range(self.ndim):
            # g = np.dot(bf_vals[:, d1, :], self.tt_cores[d1])
            g = tf.tensordot(bf_vals[:, d1, :],
                             self.tt_cores[d1], axes=[[-1], [-2]])
            mleft = tf.matmul(mleft, g)
        return tf.expand_dims(tf.squeeze(mleft), -1)


class FTTModel:
    '''
    A wrapper around FTTLayer that presents an interface
    similar to a Keras model, as well as allowing for ALS fit. The step xs->bf_vals is not trainable
    and only needs to happen once so that's mostly what this wrapper does
    '''

    def __init__(self, nodes, tt_ranks, kernel_f, **kwargs):
        self.ndim = len(tt_ranks) - 1
        self.nodes = nodes
        self.kernel_f = kernel_f
        self.tt_ranks = tt_ranks
        self.keras_model = None
        self.trainable_weights = None
        self.trainable_variables = None

        self.tt_cores = fttr.init_tt(tt_ranks, len(nodes), init_val=0.0)
        self.re_init_cores_random(for_als=True)

    def get_number_of_core_elements(self):
        n = 1
        for core in self.tt_cores:
            n += core.shape[0]*core.shape[-1]
        return n

    def re_init_cores_random(self, for_als=True):
        '''
        For whatever reason ALS works better with uniform[0,1] initialization
        and Adam with [-1,1]
        '''
        low = 0.0 if for_als else -1.0
        high = 1.0

        # try to be clever and get the order of magnitude to be about 1.0
        ncoreelems = self.get_number_of_core_elements()
        scale = (1.0/ncoreelems)**(1.0/self.ndim)
        low *= scale
        high *= scale

        for n in range(len(self.tt_cores)):
            self.tt_cores[n] = np.random.uniform(
                low=low, high=high, size=self.tt_cores[n].shape)

        # if not for_als:
        #     self.init_keras_model(**kwargs)

    def init_keras_model(self, **kwargs):
        model = keras.Sequential()
        model.add(FTTLayer(
            input_shape=(self.ndim, len(self.nodes)),
            tt_cores_init=self.tt_cores,
            **kwargs,
        ))

        model.build()
        self.keras_model = model

        # these are here to make the model look like a Keras model
        self.trainable_weights = self.keras_model.trainable_weights
        self.trainable_variables = self.keras_model.trainable_variables

        return model

    def update_from_keras_weights(self):
        self.tt_cores = copy.deepcopy(self.keras_model.get_weights())

    def bf_vals_from_xs(self, xs):
        bf_vals_list = fttr.basis_function_values(
            xs, self.nodes, self.kernel_f)
        return np.transpose(np.array(bf_vals_list), (1, 0, 2))

    def compile(self, **kwargs):
        self.keras_model.compile(**kwargs)

    def predict(self, xs, **kwargs):
        return fttr.predict(xs, self.tt_cores, self.kernel_f, self.nodes).reshape(-1, 1)

    def predict_from_keras(self, xs):
        bf_vals = self.bf_vals_from_xs(xs)
        return self.keras_model.predict(bf_vals)  # do we need to squeeze this?

    def evaluate(self, xs, ys):
        ys_fit = self.predict(xs)
        return np.linalg.norm(ys - ys_fit[:, 0])/np.linalg.norm(ys)

    def summary(self):
        self.keras_model.summary()

    def fit_als(self, xs, ys,
                rel_tol=1e-3, max_iter=10, **kwargs):
        fit_res = fttr.fit(xs, ys, self.tt_ranks, self.nodes, self.kernel_f,
                           rel_tol=rel_tol, max_iter=max_iter, init_tt_cores_val=self.tt_cores, **kwargs)

        self.tt_cores = copy.deepcopy(fit_res[0])
        return fit_res

    def fit_keras(self, xs, ys, **kwargs):
        '''
        call init_keas_model and compile first
        '''
        bf_vals = self.bf_vals_from_xs(xs)
        self.keras_model.fit(bf_vals, ys, **kwargs)
        self.update_from_keras_weights()

    # -------------------------------------------------
    # Next few methods to make FTTModel look like a Keras model
    # by delegating to self.keras_model
    # -------------------------------------------------
    def fit(self, xs, ys, for_keras=True, **kwargs):
        if for_keras:
            return self.fit_keras(xs, ys, **kwargs)
        else:
            return self.fit_als(xs, ys, **kwargs)

    def __call__(self, xs, *args, **kwargs):
        bf_vals = self.bf_vals_from_xs(xs)
        return self.keras_model(bf_vals, *args, **kwargs)
