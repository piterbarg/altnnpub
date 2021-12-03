import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from nnu import gss_kernels as gssk


def test_kernels_01(do_timings=True, tf_eager=True, tf_dtype=tf.float32, plot_figures=False):

    tf.config.run_functions_eagerly(tf_eager)

    a = 2.0
    freq = 0.5
    npts = 1000
    if do_timings:
        npts *= 10000
    x0 = 2*np.arange(npts)/npts - 1

    x = tf.constant(5*x0, dtype=tf_dtype)

    st = timer()
    lan = gssk.lanczos_kernel(a=a, freq_bound=freq)(x)
    et = timer()
    print('lancsoz timing:', et-st)

    st = timer()
    bs1 = gssk.bspline1_kernel(a=a, freq_bound=freq)(x)
    et = timer()
    print('bspline1 timing:', et-st)

    st = timer()
    bs2 = gssk.bspline2_kernel(a=a, freq_bound=freq)(x)
    et = timer()
    print('bspline2  timing:', et-st)

    st = timer()
    bs2_v1_1 = gssk.bspline2_kernel_v1_1(a=a, freq_bound=freq)(x)
    et = timer()
    print('bspline2_v1.1  timing:', et-st)

    st = timer()
    bs2_v1_2 = gssk.invquad_kernel(a=a, freq_bound=freq)(x)
    et = timer()
    print('bspline2_v1.2  timing:', et-st)

    if plot_figures:
        step = 1000
        plt.plot(x[::step], lan[::step], label='lanczos')
        plt.plot(x[::step], bs1[::step], label='bspline1')
        plt.plot(x[::step], bs2[::step], label='bspline2')
        plt.plot(x[::step], bs2_v1_1[::step], label='bspline2_v1_1')
        plt.plot(x[::step], bs2_v1_2[::step], label='invquad')
        # plt.plot(x, bs2_v2, label='bspline2_v2')
        plt.legend(loc='best')

        plt.show()
