import pandas as pd
import config
from utils.utils import load_csv


def timeseries_cv(df, sep_time):
    """
    time series cross validation
    yield X, y based on sep_time
    """
    sep_time = [pd.Timestamp(x) for x in sep_time]
    train_end = sep_time[0]
    for val_end in sep_time[1:]:
        train_idx = df.click_time <= train_end
        valid_idx = (train_end < df.click_time) & (df.click_time < val_end)
        yield train_idx, valid_idx
        train_end = val_end


def generate_cv(dir_name, fold_num, debug=None):
    """
    genarate cv dataframe in directory
    """

    for fold in range(fold_num):
        train_df = load_csv(dir_name + f'fold_{fold}_train.csv',
                            parse_dates=['click_time'],
                            n_rows=debug)
        test_df = load_csv(dir_name + f'fold_{fold}_test.csv',
                           parse_dates=['click_time'],
                           n_rows=debug)
        yield train_df, test_df
