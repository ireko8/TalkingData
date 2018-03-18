import pandas as pd
import config


def preprocess_for_nn(df, train_len=None):
    df['click_hour'] = pd.to_datetime(df.click_time).dt.hour

    train_cols = ['app', 'device', 'ip', 'os', 'channel', 'click_hour']
    embedd_config = list()

    # replace few seen values to -1 and convert continuous integers
    for col in train_cols:
        vc = df[col].value_counts()
        df.loc[df[col].isin(vc[vc <= config.UN_THR].index), col] = -1
        unique_values = sorted(df[col].unique())
        value_dim = len(unique_values)
        convert_table = dict(zip(unique_values,
                                 range(len(unique_values))))
        df[col].replace(convert_table, inplace=True)
        embedded_col_config = [col, value_dim, config.EMBEDD_DIM[col]]
        embedd_config.append(embedded_col_config)

    if train_len:
        train_X = df[:train_len, train_cols]
        train_y = df[:train_len].is_attributed
        test_X = df[train_len:, train_cols]
    else:
        train_X = df[train_cols]
        train_y = df.is_attributed
        test_X = None

    return train_X, train_y, test_X, embedd_config
