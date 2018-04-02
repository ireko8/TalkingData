from datetime import datetime
import inspect
import random
import gc
import numpy as np
import pandas as pd
import tensorflow as tf


def load_csv(fname, n_rows=None, sampling=None):
    if n_rows:
        df = pd.read_csv(fname, n_rows=n_rows)
    else:
        df = pd.read_csv(fname)

    if sampling:
        df = df.sample(sampling)

    gc.collect()
    return df


def now():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


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
