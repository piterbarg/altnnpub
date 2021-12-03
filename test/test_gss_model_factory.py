import numpy as np

# testing this module
from nnu import gss_model_factory as gssmf

# our stuff
from nnu import gss_layer, gss_report_config


def helper_generate_model_01(
    model_type=gss_report_config.ModelType.SSL,
    use_outer_regression=True,
    optimize_knots=False,
    optimize_scales=True,
    scales_dim=gss_layer.ScalesDim.OnePerKnot,
    apply_final_aggr=False,
    kernel='invquad',
    average_slopes=None,
):
    ndim = 2
    nbatch = 1000
    inputX = np.random.uniform(0.0, 1.0, size=(nbatch, ndim))
    inputY = (inputX[:, 0]-0.5)*(inputX[:, 1]-0.5)

    nknots = 100
    knots = np.random.uniform(0.0, 1.0, size=(nknots, ndim))

    nnodes_per_dim = round(pow(nknots, 1./ndim))
    global_scale = 0.5/nnodes_per_dim

    fit_model = gssmf.generate_model(
        ndim=ndim,
        global_scale=global_scale,
        nodes=knots,
        inputX=inputX,
        inputY=inputY,
        model_type=model_type,
        use_outer_regression=use_outer_regression,
        optimize_knots=optimize_knots,
        optimize_scales=optimize_scales,
        scales_dim=scales_dim,
        apply_final_aggr=apply_final_aggr,
        kernel=kernel,
        average_slopes=average_slopes
    )

    print('====================fit model======================\n\n')
    print('config:', model_type,
          use_outer_regression, optimize_knots, optimize_scales, scales_dim, apply_final_aggr)

    fit_model.summary()

    test_model = gssmf.generate_model_for_testing_2(
        fit_model, ndim=ndim, global_scale=global_scale, nodes=knots,
        model_type=model_type,
        use_outer_regression=use_outer_regression,
        optimize_knots=optimize_knots,
        optimize_scales=optimize_scales,
        scales_dim=scales_dim,
        apply_final_aggr=apply_final_aggr,
        kernel=kernel,
        average_slopes=average_slopes,
    )

    print('====================test model======================\n\n')
    test_model.summary()


def test_generate_model_01():
    helper_generate_model_01(
    )


def test_generate_model_02():
    helper_generate_model_01(
        model_type=gss_report_config.ModelType.ReLU,
    )


def test_generate_model_03():
    helper_generate_model_01(
        model_type=gss_report_config.ModelType.ReLU,
        optimize_knots=True,
    )


def test_generate_model_04():
    helper_generate_model_01(
        model_type=gss_report_config.ModelType.ReLU,
        scales_dim=gss_layer.ScalesDim.OnePerKnot,
    )


def test_generate_model_05():
    helper_generate_model_01(
        model_type=gss_report_config.ModelType.ReLU,
        scales_dim=gss_layer.ScalesDim.OnePerKnot,
    )


def test_generate_model_06():
    helper_generate_model_01(
        use_outer_regression=False,
    )


def test_generate_model_07():
    helper_generate_model_01(
        optimize_knots=True,
    )


def test_generate_model_08():
    helper_generate_model_01(
        scales_dim=gss_layer.ScalesDim.OnePerKnot,
    )


def test_generate_model_09():
    helper_generate_model_01(
        apply_final_aggr=True,
    )


def test_generate_model_10():
    helper_generate_model_01(
        scales_dim=gss_layer.ScalesDim.OnePerDim,
    )


def test_generate_model_11():
    helper_generate_model_01(
        scales_dim=gss_layer.ScalesDim.OnlyOne,
    )


def test_generate_model_12():
    helper_generate_model_01(
        scales_dim=gss_layer.ScalesDim.OnlyOne,
        average_slopes=np.array([0.5]*2)  # 2 being ndim implicitly
    )


def test_generate_model_13():
    helper_generate_model_01(
        model_type=gss_report_config.ModelType.FTT,
    )
