import pytest
import itertools
import numpy as np

from nnu import points_generator as pgen


@pytest.mark.parametrize('points_type,ndim', itertools.product(['random', 'regular', 'randomgrid'], [1, 2, 3]))
def test_01(points_type, ndim, plot_figures=False):
    nsamples = 1000

    # xs: simulating uniforms on [-range,range]^d
    sim_range = 4
    np.random.seed(314)

    nodes1d = np.linspace(-sim_range, sim_range, num=101, endpoint=True)
    x_input = pgen.generate_points(sim_range, nsamples, ndim,
                                   points_type, seed=123, nodes1d=nodes1d, plot=plot_figures)[0]

    print(f'Successfully gnerated {x_input.shape} points')
    print(f'average distance = {pgen.average_distance(x_input)}')
