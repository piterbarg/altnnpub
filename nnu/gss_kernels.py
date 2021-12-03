import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from timeit import default_timer as timer


def global_kernel_dict(global_scale, tf_dtype=tf.float32):
    return {
        'sinc': sinc_kernel(freq_bound=1.0/global_scale, tf_dtype=tf_dtype),
        'lanczos': lanczos_kernel(a=2.0, freq_bound=1.0/global_scale, tf_dtype=tf_dtype),
        'bspline1': bspline1_kernel(a=2.0, freq_bound=1.0/global_scale, tf_dtype=tf_dtype),
        'bspline2': bspline2_kernel(a=2.0, freq_bound=1.0/global_scale, tf_dtype=tf_dtype),
        'invquad': invquad_kernel(a=2.0, freq_bound=1.0/global_scale, tf_dtype=tf_dtype),
        'bspline2_v2': bspline2_kernel_v2(a=2.0, freq_bound=1.0/global_scale, tf_dtype=tf_dtype),
    }


def tfsinc(x, tf_dtype=tf.float32):
    eps = 1e-4
    return tf.cast(K.abs(x) > eps, tf_dtype)*tf.math.divide_no_nan(K.sin(K.constant(math.pi)*x), K.constant(math.pi)*x) \
        + tf.cast(K.abs(x) <= eps, tf_dtype)


def sinc_kernel(freq_bound=1, tf_dtype=tf.float32):
    return lambda x: tfsinc(x*freq_bound, tf_dtype)


def lanczos_kernel(a=2.0, freq_bound=1, tf_dtype=tf.float32):
    def lanczos_kernel_(x):
        return tf.cast(K.abs(x) <= a, tf_dtype)*tfsinc(x, tf_dtype)*tfsinc(x/a, tf_dtype)
    return lambda x: lanczos_kernel_(x*freq_bound)


def invquad_kernel(a=2.0, freq_bound=1, tf_dtype=tf.float32):
    def invquad_kernel_(x):
        xs = (x*freq_bound/a) * 3.0/2.0
        return K.maximum(1.15 / (1 + xs*xs/0.35)-0.15, 0.0)
    return invquad_kernel_


def bspline1_kernel(a=2.0, freq_bound=1, tf_dtype=tf.float32):
    def bspline_1_(x):
        return K.maximum(1 - K.abs(x/a), 0.0)
    return lambda x: bspline_1_(x*freq_bound)


def bspline2_kernel(a=2.0, freq_bound=1, tf_dtype=tf.float32):

    def B1(x):
        return tf.cast(x >= 0, tf_dtype)*tf.cast(x < 1, tf_dtype)*x*x/2

    def B2(x):
        return tf.cast(x >= 1, tf_dtype)*tf.cast(x < 2, tf_dtype)*(-2*x*x+6*x-3)/2

    def B3(x):
        return tf.cast(x >= 2, tf_dtype)*tf.cast(x < 3, tf_dtype)*(3-x)**2/2

    def bspline_2_(x):
        xs = (x*freq_bound/a+1)*3.0/2.0
        return (B1(xs) + B2(xs) + B3(xs))/0.75

    return bspline_2_


def bspline2_kernel_v1_1(a=2.0, freq_bound=1, tf_dtype=tf.float32):

    def B1(x, x2):
        return x2/2/0.75

    def B2(x, x2):
        return (-2*x2+6*x-3)/2/0.75

    def B3(x, x2):
        return (9-6*x+x2)/2/0.75

    def bspline_1_1_(x):
        xs = (x*freq_bound/a+1)*3.0/2.0
        xs2 = xs*xs
        ind0 = tf.cast(xs >= 0, tf_dtype)
        ind1 = tf.cast(xs >= 1, tf_dtype)
        ind2 = tf.cast(xs >= 2, tf_dtype)
        ind3 = tf.cast(xs >= 3, tf_dtype)

        b1 = B1(xs, xs2)
        b2 = B2(xs, xs2)
        b3 = B3(xs, xs2)
        return b1*ind0 + (b2-b1) * ind1 + (b3 - b2) * ind2 - b3*ind3

    return bspline_1_1_


def bspline2_kernel_v2(a=2.0, freq_bound=1, tf_dtype=tf.float32):

    # scalar
    @ tf.function
    def bspline_2_v2_(x):
        xs = (x*freq_bound/a+1)*3.0/2.0

        # @tf.function makes no difference
        def B1(x):
            return x*x/2

        # @tf.function makes no difference
        def B2(x):
            return (-2*x*x+6*x-3)/2

        # @tf.function makes no difference
        def B3(x):
            return (3-x)**2/2

        if xs < 0.0:
            return 0.0
        if xs < 1.0:
            return B1(xs)/0.75
        if xs < 2.0:
            return B2(xs)/0.75
        if xs < 3.0:
            return B3(xs)/0.75
        return 0.0

    @ tf.function
    def bspline_2_v2_tf_(x):
        return tf.map_fn(fn=bspline_2_v2_, elems=x)

    return bspline_2_v2_tf_
