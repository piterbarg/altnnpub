import pytest
import pandas as pd

# our stuff
from nnu import gss_report_generator as gssgen
from nnu import gss_report_config as gssrconfig


@pytest.mark.parametrize(
    'model_name,run_id,expected_dict',
    [
        ('onedim_regr', '01_250_1',
         {
             'learn_mse': 0.1738689167528844,
             'learn_mae': 0.08392812095129284,
             'fitting_time': 2.979267,
             'n_params': 1,
             'n_epochs': 0.025,
             'mc_error': -0.009239387197101712,
             'test_mse': 0.033212632153750025,
             'test_mse_stdev': 0.003562413038877414,
             'test_mae': 0.009488255642174653,
             'input_f_spec': 'laplace_0',
             'ndim': 3,
             'l2_regularizer': 1e-06,
             'nsr_stretch': 1.0,
             'kernel': 'invquad',
             'nsamples': 10000,
             'model': 'onedim_regr',
             'seed_keras': 2021,
             'seed_input': 1917,
             'batch_label': '',
             'testing_time': 6.784382599999994,
             'run_id': '01_250_1'
         }),
        ('ftt_als', '04_500_1',
         {
             'learn_mse': 0.026291913811857078,
             'learn_mae': 0.011663415917845083,
             'fitting_time': 2.353505499999999,
             'n_params': 1750,
             'n_epochs': 5,
             'mc_error': -0.009239387197101712,
             'test_mse': 0.0267953071447491,
             'test_mse_stdev': 0.0011861304325814807,
             'test_mae': 0.012222374058009753,
             'input_f_spec': 'laplace_0',
             'ndim': 3,
             'l2_regularizer': 1e-16,
             'nsr_stretch': 1.0,
             'kernel': 'invquad',
             'nsamples': 10000,
             'model': 'ftt_als',
             'seed_keras': 2021,
             'seed_input': 1917,
             'batch_label': '',
             'testing_time': 0.7723028999999997,
             'run_id': '04_500_1'
         }),
    ]
)
def test_model_numbers_01(model_name, run_id, expected_dict):
    nsamples = 10000

    # select which function to fit
    input_f_spec, ndim = 'laplace_0', 3

    # which kernel
    kernel = 'invquad'

    run_names = [run_id]
    short_run = False

    # Either specify which models to run or set to None to run them all
    model_names = [model_name]

    # How many test runs -- to get test_mse_stdev for example
    test_res_runs = 10

    # Want to see figures?
    show_figures = False

    # Set up the batch
    batch_spec = gssrconfig.get_standard_batch_specs(
        ndim=ndim, nsamples=nsamples, kernel=kernel,
        input_f_spec=input_f_spec, test_res_runs=test_res_runs)
    batch_spec['model_names'] = model_names
    batch_spec['run_names'] = run_names
    batch_spec['show_figures'] = show_figures

    # ... and run it
    actual_df, _ = gssgen.generate_results_all_runs(
        short_run=short_run,
        file_name=None,
        batch_spec=batch_spec,
    )

    expected_df = pd.DataFrame.from_records([expected_dict])

    # remove hardware dependent values
    actual_df.drop(columns=['fitting_time', 'testing_time'], inplace=True)
    expected_df.drop(columns=['fitting_time', 'testing_time'], inplace=True)
    pd.testing.assert_frame_equal(
        actual_df, expected_df, check_exact=False, rtol=1e-3, atol=1e-3)
