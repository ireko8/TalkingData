import pandas as pd
import config


def proc_all(df):
    """
    process for all of the data
    we limit ip range under config.IP_MAX
    """
    df['click_hour'] = df.click_time.dt.hour
    df = df.drop(['attributed_time'],
                 axis=1)
    df = df[df.ip <= config.IP_MAX]
    return df


def make_min_leaf(train_df, test_df, logger):
    """
    aggregate few vals into -1
    """
    for col in config.TRAIN_COLS:

        uniq_vals = train_df[col].unique()
        test_df[col].apply(lambda x: x if x in uniq_vals else -1)
        vc = train_df[col].value_counts()
        few_vals = vc[vc <= config.UN_THR[col]].index
        train_df.loc[train_df[col].isin(few_vals), col] = -1
        test_df.loc[test_df[col].isin(few_vals), col] = -1

    return train_df, test_df
        

def make_embedded_conf(train_df, test_df, logger):
    """
    configure embedded size for each column of data
    and convert data to the number corresponding to embedding
    """
    
    embedd_config = list()

    for col in config.EMBEDD_DIM.keys():

        uniq_vals = sorted(train_df[col].unique())
        value_dim = len(uniq_vals)
        convert_table = dict(zip(uniq_vals,
                                 range(len(uniq_vals))))
        train_df[col].replace(convert_table, inplace=True)
        test_df[col].replace(convert_table, inplace=True)
        
        embedded_col_config = [col,
                               value_dim,
                               config.EMBEDD_DIM[col]]
        
        embedd_config.append(embedded_col_config)

    return train_df, test_df, embedd_config


def make_onehot(train_df, test_df, logger):
    train_len = train_df.shape[0]
    data = pd.concat([train_df, test_df])
    logger.info(data.columns)
    with logger.interval_timer("OneHot"):
        data = pd.get_dummies(data, columns=config.TRAIN_COLS)

    conf = list(data.columns)
    conf.remove('is_attributed')
    logger.info(data.columns)
    return data[:train_len], data[train_len:], conf


def proc_emb_nn(train_df, test_df, logger):
    """
    preprocess for embedding NN
    """
    train_df, test_df = make_min_leaf(train_df, test_df, logger)
    return make_embedded_conf(train_df, test_df, logger)


def proc_sparse_nn(train_df, test_df, logger):
    """
    preprocess for sparse NN
    """
    train_df, test_df = make_min_leaf(train_df, test_df, logger)
    return make_onehot(train_df, test_df, logger)


def proc_xgb(train_df, test_df, cols, logger):
    # return make_min_leaf(train_df, test_df, logger)
    return make_count_encoding(train_df, test_df, cols, logger)
