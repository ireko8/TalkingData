from contextlib import contextmanager
import pandas as pd
import numpy as np


@contextmanager
def ct_indexing(df):
    try:
        df.index = df.click_time
        yield df
    finally:
        df.reset_index()


def make_td(df, cols, logger):
    """ make first flag for given columns
    """
    df.click_time = pd.to_datetime(df.click_time)
    for c in cols:
        ct = "click_time"
        fc_col = f"{c}_ft"
        td_col = f"{c}_td"
        grouped = df.groupby(c, as_index=False)
        first_click = grouped.click_time.min()
        logger.info(first_click.columns)
        first_click = first_click.rename(columns={'click_time': fc_col})
        df = pd.merge(df, first_click, on=c)
        df[td_col] = (df[ct] - df[fc_col]).astype("timedelta64[s]")
        df = df.drop([fc_col], axis=1)

    df.reset_index()
    return df


def make_sequence(df, cols):
    """ make sequence for given cols
    """
    NotImplemented


def make_cross(df, cols):
    NotImplemented
