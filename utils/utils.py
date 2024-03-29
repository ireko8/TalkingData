from datetime import datetime
import inspect
import random
import gc
import numpy as np
import pandas as pd
import dask.dataframe as dd
# import tensorflow as tf


def load_csv(fname,
             n_rows=None,
             sampling=None,
             dtypes=None,
             parse_dates=None,
             compute=True):
    if n_rows:
        df = pd.read_csv(fname,
                         dtype=dtypes,
                         nrows=n_rows,
                         parse_dates=parse_dates)
    else:
        df = dd.read_csv(fname,
                         dtype=dtypes,
                         blocksize=2560000,
                         parse_dates=parse_dates)
        if compute:
            df = df.compute()
    gc.collect()
    return df


def load_feather(fname,
                 n_rows=None,
                 parse_dates=None):
    if n_rows:
        df = pd.read_feather(fname,
                             nrows=n_rows,
                             parse_dates=parse_dates)
    else:
        df = pd.read_feather(fname)
    gc.collect()
    return df


def now():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed)


def df_to_list(df):
    return [df[c] for c in df.columns]


def df_to_onehot(df):
    return pd.get_dummies(df)


def dump_module(mod_name):
    return inspect.getsource(mod_name)


def rank_average(prob_list):
    """todo: examine performance of rankdata and np.argsort
    """
    probs = np.array(prob_list)
    return np.argsort(probs)/probs.shape[0]
