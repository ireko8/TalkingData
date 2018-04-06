from contextlib import contextmanager
import pandas as pd
import dask.dataframe as dd


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
    for c in cols:
        ct = "click_time"
        fc_col = f"{c}_ft"
        td_col = f"{c}_td"
        
        first_click = df.groupby(c).click_time.min().reset_index()
        logger.info(first_click.columns)
        first_click = first_click.rename(columns={'click_time': fc_col})
        df = dd.merge(df, first_click, on=c)
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


def make_count_encoding(df, cols, logger):
    """ count encoding for each columns
    """
    for col in cols:
        ce_col = f"{col}_ce"
        vc = df[col].value_counts()
        df[ce_col] = df[col].apply(lambda x: vc[x])

    return df


def make_rolling(df, cols, logger, window_size):
    """ rolling mean for each cols
    """
    for col in cols:
        rm_col = f"{col}_rm"
        df[rm_col] = df[col].rolling(window_size).mean()

    return df
