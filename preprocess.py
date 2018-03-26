import pandas as pd
import config


def proc_all(df):
    """
    process for all of the data
    """
    df['click_hour'] = pd.to_datetime(df.click_time).dt.hour
    return df


def make_embedded_conf(train_df, test_df, logger):
    """
    configure embedded size for each column of data
    and convert data to the number corresponding to embedding
    """
    
    embedd_config = list()

    for col in config.TRAIN_COLS:

        uniq_vals = train_df[col].unique()
        test_df[col].apply(lambda x: x if x in uniq_vals else -1)
        vc = train_df[col].value_counts()
        few_vals = vc[vc <= config.UN_THR[col]].index
        train_df.loc[train_df[col].isin(few_vals), col] = -1
        test_df.loc[test_df[col].isin(few_vals), col] = -1
        
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
    

def proc_emb_nn(train_df, test_df, logger):

    return make_embedded_conf(train_df, test_df, logger)
