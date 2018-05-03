import gc
from pathlib import Path
from pprint import pformat
import pandas as pd
from lightgbm import LGBMClassifier
import config
from preprocess import CustomTargetEncoder
from validation import timeseries_cv
from utils.logger import Logger
from utils.utils import set_seed, now, load_csv, load_feather


def extract_key(path):
    """ extract feature key for merge
    """
    p = Path(path).stem
    key = str(p).split('_')[:-1]
    return key
    

def load_data(base, features, logger, dump=None):
    with logger.interval_timer('base load'):
        df = load_csv(base,
                      dtypes=config.TRAIN_DTYPES,
                      parse_dates=config.TRAIN_PARSE_DATES)
        df['hour'] = df.click_time.dt.hour
        df['day'] = df.click_time.dt.day
        df = df.reset_index(drop=True)
        if 'attributed_time' in df.columns:
            df = df.drop('attributed_time', axis=1)
            
    with logger.interval_timer('features'):
        fs = [load_csv(p).reset_index(drop=True) for p in features]
        df = df.join(fs)

    if dump:
        df.to_feather(dump)
    return df


def test_sup_merge(test_supplement):
    test = load_csv('input/test.csv',
                    dtypes=config.TEST_DTYPES,
                    parse_dates=config.TEST_PARSE_DATES)
    join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    all_cols = join_cols + ['is_attributed']
    test = test.merge(test_supplement[all_cols], how='left', on=join_cols)
    test = test.drop_duplicates(subset=['click_id'])
    return test


def proc_bf_cv(df):
    cat_cols = ['app', 'ip', 'device', 'hour', 'os', 'channel']
    td_cols = [c for c in df.columns if 'td' in c]
    df[td_cols] = df[td_cols] == 0
    cat_cols += td_cols
    for c in cat_cols:
        df[c] = df[c].astype('category')
    return df


def custom_encode(train_df, valid_df, encode_list, threshold, logger):
    """ target encoding for given df and list
    """
    tes = []
    for e in encode_list:
        logger.info(e)
        col_name = '_'.join(e)
        te_name = col_name + '_te'
        te = CustomTargetEncoder(threshold)
        te.fit(train_df, e, 'is_attributed', te_name)
        train_df = te.transform(train_df)
        valid_df = te.transform(valid_df)
        tes.append(te)

    return train_df, valid_df, tes


def experiment(train=None,
               test=None):
    """experiment func
    """
    
    cv_name = now()
    cv_log_path = f'cv/LightGBM/{cv_name}/'
    Path(cv_log_path).mkdir(parents=True, exist_ok=True)
    
    log_fname = cv_log_path + 'cv.log'
    cv_logger = Logger('CV_log', log_fname)
    cv_logger.info("Experiment Start")

    with cv_logger.interval_timer('load data'):
        if train:
            train_df = load_feather(train)
        else:
            fs = Path('preprocessed/features').glob('train_*.csv')
            # fs = ['preprocessed/features/train_nextClick.csv',
            #       'preprocessed/features/train_ip_app_nextClick.csv']
            train_df = load_data(config.TRAIN_PATH, fs, cv_logger,
                                 dump='preprocessed/train.ftr')
        offset = pd.to_datetime('2017-11-07 16:00:00')
        train_df = train_df[train_df.click_time >= offset]
        gc.collect()
        if test:
            test_df = load_feather(test)

        else:
            fs = Path('preprocessed/features').glob('test_*.csv')
            # fs = ['preprocessed/features/test_nextClick.csv',
            #       'preprocessed/features/test_ip_app_nextClick.csv']
            test_df = load_data(config.TEST_PATH, fs, cv_logger,
                                dump='preprocessed/test.ftr')
            gc.collect()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    with cv_logger.interval_timer('split'):
        split_gen = enumerate(timeseries_cv(train_df,
                                            config.SEP_TIME))

    # dump configuration
    aucs = []

    # add ip_day_hour_nunique, app_device_channel_nextClick,
    # ip_os_device_nextClick,
    train_cols = ['app',
                  'app_channel_ce',
                  'channel',
                  'device',
                  'hour',
                  'ip_app_ce',
                  'ip_app_channel_hour_mean',
                  'ip_app_channel_nextClick',
                  'ip_app_device_os_channel_nextClick',
                  'ip_app_device_os_nextClick',
                  'ip_app_nextClick',
                  'ip_app_nunique',
                  'ip_app_os_ce',
                  'ip_app_os_nunique',
                  'ip_channel_nunique',
                  'ip_day_hour_ce',
                  'ip_day_nunique',
                  'ip_day_hour_nunique',
                  'ip_device_nunique',
                  'ip_device_os_app_cumcount',
                  'ip_nextClick',
                  'ip_os_device_nextClick',
                  'app_device_channel_nextClick',
                  'ip_os_cumcount',
                  'ip_os_device_app_nunique',
                  'os']
    # encode_list = config.ENCODE_LIST
    # threshold = config.TE_THR

    valid_time = [4, 5, 6, 9, 10, 11, 13, 14, 15]
    # public_time = [5, 6, 9, 10, 11, 13, 14, 15]
    train_df = proc_bf_cv(train_df)
    gc.collect()
    
    for num, (train_idx, valid_idx) in split_gen:
        cv_logger.kiritori()
        cv_logger.info(f"fold {num} start")

        with cv_logger.interval_timer('train test split'):
            cvtrain_df = train_df.loc[train_idx]
            valid_df = train_df.loc[valid_idx]
            valid_df2 = valid_df[valid_df.hour.isin(valid_time)]
            # valid_df3 = valid_df[valid_df.hour == 4]
            # valid_df4 = valid_df[valid_df.hour.isin(public_time)]
            # with cv_logger.interval_timer('target encode'):
            # cvtrain_df, valid_df, tes = custom_encode(cvtrain_df,
            #                                           valid_df,
            #                                           encode_list,
            #                                           threshold,
            #                                           cv_logger)
            # cvtrain_df = proc_bf_cv(cvtrain_df)
            # valid_df = proc_bf_cv(valid_df)
            # train_cols += [c for c in cvtrain_df.columns if '_te' in c]
            
        cv_logger.info("LGBM Baseline validation")

        eval_names = ['train',
                      # 'valid_all',
                      # 'valid_pub',
                      # 'valid_priv',
                      'valid_lb']
        train_X, train_y = cvtrain_df[train_cols], cvtrain_df.is_attributed
        eval_set = [(train_X, train_y)]
        with cv_logger.interval_timer('valid make'):
            for df in [valid_df2]:
                X, y = df[train_cols], df.is_attributed
                eval_set.append((X, y))

            cv_logger.info(list(train_X.columns))
            gc.collect()
            
        lgbm = LGBMClassifier(n_estimators=1000,
                              learning_rate=0.1,
                              num_leaves=31,
                              max_depth=-1,
                              min_child_samples=20,
                              min_child_weight=5,
                              max_bin=255,
                              scale_pos_weight=200,
                              colsample_bytree=0.3,
                              subsample=0.6,
                              subsample_freq=0,
                              n_jobs=24)

        cv_logger.info(lgbm.get_params())
        lgbm.fit(train_X,
                 train_y,
                 eval_metric="auc",
                 eval_set=eval_set,
                 eval_names=eval_names,
                 early_stopping_rounds=30,
                 verbose=10)
        auc = lgbm.best_score_
        aucs.append(auc)

        cv_logger.info(f"naive LGBM AUC : {auc}")
        cv_logger.info(pformat(lgbm.evals_result_))

        cv_logger.info("feature importance")
        fi = dict(zip(train_X.columns, lgbm.feature_importances_))
        cv_logger.info(pformat(fi))
        cv_logger.info(f"fold {num} end")

    del train_df
    cv_logger.double_kiritori()
    cv_logger.info("Cross Validation Done")
    cv_logger.info("Naive LGBM")
    cv_logger.info(f"AUC {auc}")

    cv_logger.info("Predict")

    # with cv_logger.interval_timer('all target encode'):
    #     for te in tes:
    #         test_df = te.transform(test_df)

    test_df = proc_bf_cv(test_df)

    test_X = test_df[train_cols]
    pred = lgbm.predict_proba(test_X,
                              num_iteration=lgbm.best_iteration_)
    test_df['is_attributed'] = pred[:, 1]
    test_df['click_id'] = test_df.click_id.astype('uint32')
    sub = test_sup_merge(test_df)
    sub[['click_id', 'is_attributed']].to_csv(f'sub/{cv_name}.csv',
                                              index=False)
    
    cv_logger.info("Experiment Done")


if __name__ == '__main__':
    set_seed(2018)
    experiment(train='preprocessed/train.ftr',
               test='preprocessed/test.ftr')
