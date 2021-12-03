import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime


def set_seed_for_keras(seed=31415926):
    """
    Set all the seeds needed, for results reproducibility, see
    https://stackoverflow.com/a/52897289/14551426
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    # for later versions:
    # tf.compat.v1.set_random_seed(seed)


def get_nonexistent_path(fname_path: str, datestamp: str = ''):
    """
    Get the path to a filename which does not exist by incrementing path.
    adapted from https://stackoverflow.com/a/43167607

    Examples with empty datestamp
    --------
    >>> get_nonexistent_path('/etc/issue')
    '/etc/issue-001'
    >>> get_nonexistent_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    filename, file_extension = os.path.splitext(fname_path)
    if datestamp:
        filename = f'{filename}-{datestamp}'

    fname_path = f'{filename}{file_extension}'
    if not os.path.exists(fname_path):
        return fname_path

    i = 1
    while True:
        new_fname = f'{filename}-{i:03d}{file_extension}'
        if not os.path.exists(new_fname):
            return new_fname
        i += 1


def df_save_to_results(df: pd.DataFrame, file_name: str, suffix: str = '', add_date=True):
    if file_name is None:
        return

    if suffix:
        file_base, file_extension = os.path.splitext(file_name)
        file_base += '_'
        file_base += suffix
        file_name = file_base + file_extension

    nnu_python_nnu_folder = os.path.dirname(os.path.abspath(__file__))
    nnu_python_folder = os.path.join(nnu_python_nnu_folder, os.path.pardir)
    nnu_results_folder = os.path.join(nnu_python_folder, 'results')
    results_file = os.path.join(nnu_results_folder, file_name)

    datestring = ''
    if add_date:
        filename_date_fmt = '%Y%m%d'
        today = datetime.now()
        datestring = today.strftime(filename_date_fmt)

    results_file = get_nonexistent_path(
        results_file, datestamp=datestring)
    df.to_csv(results_file)
