from enum import Enum
import tensorflow as tf

# our stuff
from nnu import gss_layer


class ModelType(Enum):
    SSL = 0
    FTT = 1
    ReLU = 2


def get_standard_batch_specs(
    ndim=3,
    nsamples=10000,
    kernel='lanczos',
    input_f_spec: str = 'laplace_0',
    batch_label='',
    test_res_runs=1,
):
    batch_spec = dict(
        ndim=ndim,
        nsamples=nsamples,
        kernel=kernel,
        input_f_spec=input_f_spec,
        seed_for_keras=2021,
        input_seed=1917,
        test_res_seed=314,
        test_res_runs=test_res_runs,
        show_figures=False,
        l2_multiplier=1,
        tf_dtype=tf.float32,
        nsr_stretch=1.3,
        model_names=None,
        run_names=None,
        batch_label=batch_label,
        generate_more_nodes=True,
        sim_range=4.0,
    )

    update_batch_spec_for_input_f_spec(batch_spec=batch_spec)

    return batch_spec


def update_batch_spec_for_input_f_spec(batch_spec):
    '''
    we have somewhat complicated logic for figuring out nsr_stretch
    This method needs to be called every time we change input_f_spec in batch_spec
    It modifies the arg in place
    '''
    batch_spec['nsr_stretch'] = 1.3

    if batch_spec['input_f_spec'].startswith('laplace'):
        batch_spec['nsr_stretch'] = 1.0
    return batch_spec


def get_model_specs(kernel='lanczos'):

    model_specs = {
        'relu': dict(
            model_type=ModelType.ReLU,
            use_outer_regression=False,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnePerKnot,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-6,  # not relevant
        ),
        'base': dict(
            model_type=ModelType.SSL,
            use_outer_regression=False,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnePerKnot,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-6,  # not relevant
        ),
        'hidim_regr': dict(
            model_type=ModelType.SSL,
            use_outer_regression=True,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnePerKnot,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-6,
        ),
        'lodim_regr': dict(
            model_type=ModelType.SSL,
            use_outer_regression=True,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnePerDim,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-6,
        ),
        'hidim_regr_bfgs': dict(
            model_type=ModelType.SSL,
            use_outer_regression=True,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnePerKnot,
            apply_final_aggr=True,
            kernel=kernel,
            l2_regularizer=1e-6,
        ),
        'lodim_regr_bfgs': dict(
            model_type=ModelType.SSL,
            use_outer_regression=True,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnePerDim,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-6,
        ),
        'onedim_regr_bfgs': dict(
            model_type=ModelType.SSL,
            use_outer_regression=True,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnlyOne,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-6,
        ),
        'onedim_regr': dict(
            model_type=ModelType.SSL,
            use_outer_regression=True,
            optimize_knots=False,
            optimize_scales=True,
            scales_dim=gss_layer.ScalesDim.OnlyOne,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-6,
        ),
        'ftt': dict(
            model_type=ModelType.FTT,
            use_outer_regression=False,
            optimize_knots=False,
            optimize_scales=False,
            scales_dim=gss_layer.ScalesDim.OnePerDim,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-16,
        ),
        'ftt_als': dict(
            model_type=ModelType.FTT,
            use_outer_regression=False,
            optimize_knots=False,
            optimize_scales=False,
            scales_dim=gss_layer.ScalesDim.OnePerKnot,
            apply_final_aggr=False,
            kernel=kernel,
            l2_regularizer=1e-16,
        ),
    }
    return model_specs


def get_optimizer_specs():
    opt_specs = {
        'relu': dict(
            type='adam',
            epoch_mult=100,
            # epoch_mult=1,
            learn_rate=0.001,
            is_regr=False,
        ),
        'base': dict(
            type='adam',
            epoch_mult=3.5,
            learn_rate=0.01,
            is_regr=False,
        ),
        'hidim_regr': dict(
            type='adam',
            epoch_mult=1,
            learn_rate=0.1,
            is_regr=True,
        ),
        'lodim_regr': dict(
            type='adam',
            epoch_mult=1,
            learn_rate=0.1,
            is_regr=True,
        ),
        'hidim_regr_bfgs': dict(
            type='bfgs',
            epoch_mult=0.1,
            learn_rate=1.0,
            is_regr=True,
        ),
        'lodim_regr_bfgs': dict(
            type='bfgs',
            epoch_mult=0.05,
            learn_rate=1.0,
            is_regr=True,
        ),
        'onedim_regr_bfgs': dict(
            type='bfgs_1d',
            epoch_mult=0.025,
            learn_rate=1.0,
            is_regr=True,
        ),
        'onedim_regr': dict(
            type='1d',
            epoch_mult=0.025,
            learn_rate=1.0,
            is_regr=True,
        ),
        'ftt': dict(
            type='adam',
            epoch_mult=30,
            learn_rate=0.005,  # 0.02,
            is_regr=False,
        ),
        'ftt_als': dict(
            type='als',
            epoch_mult=5,
            learn_rate=1.25,  # 1.0,
            is_regr=False,
        ),
    }
    return opt_specs


def get_run_specs(nnodes=250, epochs=1, short_run=False, get_user_defined=False):
    if get_user_defined:
        return {
            'default': dict(
                epochs=epochs,
                nnodes=nnodes,
            ),
        }

    run_specs = {
        '01_250_1': dict(
            epochs=1,
            nnodes=250,
        ),
        '02_250_2': dict(
            epochs=2,
            nnodes=250,
        ),
    }
    if not short_run:
        run_specs.update({
            '03_250_3': dict(
                epochs=3,
                nnodes=250,
            ),
            '04_500_1': dict(
                epochs=1,
                nnodes=500,
            ),
            '05_500_2': dict(
                epochs=2,
                nnodes=500,
            ),
            '06_500_3': dict(
                epochs=3,
                nnodes=500,
            ),
            '07_1000_1': dict(
                epochs=1,
                nnodes=1000,
            ),
            '08_1000_2': dict(
                epochs=2,
                nnodes=1000,
            ),
            '09_1000_3': dict(
                epochs=3,
                nnodes=1000,
            ),

        })
    return run_specs
