from timeit import default_timer as timer

# our stuff
from nnu import gss_report_generator as gssgen
from nnu import gss_report_config as gssrconfig


def test_generate_results_01(kernel='lanczos'):

    batch_spec = gssrconfig.get_standard_batch_specs(kernel=kernel)
    batch_spec['show_figures'] = True
    batch_spec['test_res_runs'] = 1
    batch_spec['ndim'] = 2
    batch_spec['nsamples'] = 10
    return gssgen.generate_results_all_models(batch_spec=batch_spec,)


def test_gen_res_all_runs_01(batch_spec=None):

    if batch_spec is None:
        batch_spec = gssrconfig.get_standard_batch_specs(
            kernel='bspline1')
    batch_spec['nsamples'] = 10

    res_df, _ = gssgen.generate_results_all_runs(
        short_run=True,
        batch_spec=batch_spec
    )
    print(res_df)


def run_single_batch(
        batch_spec=None, testing_only=False, short_run=True):

    if testing_only:
        test_gen_res_all_runs_01(
            batch_spec=batch_spec,
        )

    else:

        if batch_spec is None:
            batch_spec = gssrconfig.get_standard_batch_specs()

        gssgen.generate_results_all_runs(
            short_run=short_run,
            file_name='ss_latest.csv',
            batch_spec=batch_spec,
        )


def do_adhoc_run():
    nsamples = 10000

    # select which function to fit
    # input_f_spec, ndim = 'laplace_0', 3
    # input_f_spec, ndim = 'laplace_1', 3
    # input_f_spec, ndim = 'midrank_0', 5
    input_f_spec, ndim = 'midrank_2', 5
    # input_f_spec, ndim = 'doput_0', 5
    # input_f_spec, ndim = 'doput_1', 5
    # input_f_spec, ndim = 'midrank_0', 11
    # input_f_spec, ndim = 'midrank_5', 11

    # (or loop over all of them)
    input_fs = [('laplace_0', 3), ('laplace_1', 3),
                ('midrank_0', 5), ('midrank_2', 5),
                ('doput_0', 5), ('doput_1', 5),
                ('midrank_0', 11), ('midrank_5', 11),
                ]

    # which kernel
    kernel = 'invquad'

    # Either specify specific run_ids or set to None to do them all
    # (Full list in gss_report_config.py)
    # run_names = ['01_250_1', '04_500_1', '07_1000_1']
    run_names = None
    # or simply take the first two runs by setting this to True
    short_run = True

    # Either specify which models to run or set to None to run them all
    # (Full list in gss_report_config.py)
    # model_names = ['onedim_regr', 'ftt_als', ]
    model_names = None

    # How many test runs -- to get test_mse_stdev for example
    test_res_runs = 1

    # For quick checks can set testing_only = True
    testing_only = False

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
    run_single_batch(
        batch_spec=batch_spec, testing_only=testing_only, short_run=short_run)


if __name__ == '__main__':

    start_time = timer()
    do_adhoc_run()
    end_time = timer()

    run_time = end_time - start_time
    print(f'Run finished in {run_time} seconds')
