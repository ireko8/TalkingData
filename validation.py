import config


def time_series_cv(df, sep_time):
    """
    time series cross validation
    yield X, y based on sep_time
    """
    train_end = sep_time[0]
    for val_end in sep_time[1:]:
        train_idx = df.click_time <= train_end
        valid_idx = (train_end < df.click_time) & (df.click_time < val_end)
        yield train_idx, valid_idx
        train_end = val_end


