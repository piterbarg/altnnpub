import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from timeit import default_timer as timer

from tensorflow.keras import activations

from nnu import gss_kernels as gssk


class ScalesDim(Enum):
    OnePerKnot = 0
    OnePerDim = 1
    OnlyOne = 2


class ProdKernelLayer(keras.layers.Layer):

    def __init__(self,
                 knots,
                 scales=None,
                 optimize_knots=False,
                 optimize_scales=False,
                 scales_dim=ScalesDim.OnePerKnot,
                 activation=gssk.lanczos_kernel(),
                 average_slopes=None,
                 **kwargs):

        super(ProdKernelLayer, self).__init__(**kwargs)

        self.knots_init = knots
        self.scales_init = scales
        self.nknots = knots.shape[0]
        self.ndim = knots.shape[1]
        self.optimize_knots = optimize_knots
        self.optimize_scales = optimize_scales
        self.scales_dim = scales_dim
        self.activation = activations.get(activation)

        if average_slopes is not None:
            self.average_slopes = np.reshape(average_slopes, (1, self.ndim))
        else:
            self.average_slopes = np.ones((1, self.ndim))

    def build(self, input_shape):
        if input_shape[1] != self.ndim:
            raise ValueError('Incompatible dimensions')

        if self.scales_init is not None:
            if self.scales_dim == ScalesDim.OnePerKnot and self.scales_init.shape != self.knots_init.shape:
                raise ValueError(
                    f'Incompatible dimensions for scales with option {self.scales_dim}, expecting {self.knots_init.shape}')
            elif self.scales_dim == ScalesDim.OnePerDim and self.scales_init.shape != (1, self.ndim):
                raise ValueError(
                    f'Incompatible dimensions for scales with option {self.scales_dim}, expecting {(1,self.ndim)}')
            elif self.scales_dim == ScalesDim.OnlyOne and self.scales_init.shape != (1, 1):
                raise ValueError(
                    f'Incompatible dimensions for scales with option {self.scales_dim}, expecting {(1,1)}')

        self.knots = self.add_weight(
            'knots',
            shape=[self.nknots, self.ndim],
            dtype=self.dtype,
            trainable=self.optimize_knots
        )
        self.knots.assign(self.knots_init)

        self.n_scales_per_dim = 1 if self.scales_dim != ScalesDim.OnePerKnot else self.knots.shape[
            0]
        self.sdim = self.ndim if self.scales_dim != ScalesDim.OnlyOne else 1
        self.scales = self.add_weight(
            'scales',
            shape=[self.n_scales_per_dim, self.sdim],
            dtype=self.dtype,
            trainable=self.optimize_scales
        )

        if self.scales_init is not None:
            self.scales.assign(self.scales_init)
        else:
            self.scales.assign(0.5*np.ones((self.n_scales_per_dim, self.sdim)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nknots, self.ndim)

    def call(self, x, mask=None):

        x1 = tf.tile(tf.expand_dims(x, axis=1), [1, self.nknots, 1])
        x2 = (x1 - self.knots)*tf.tile(self.scales * self.average_slopes,
                                       (self.knots.shape[0]//self.n_scales_per_dim, 1))

        return self.activation(x2)
