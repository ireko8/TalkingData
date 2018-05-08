import gc
from argparse import ArgumentParser
from pathlib import Path
import multiprocessing as mp
import numpy as np
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
import ipdb
import config
from validation import timeseries_cv, generate_cv
from utils.logger import Logger
from utils.utils import load_feather, set_seed, now


def to_parallel(df, process, groups):
    pool = mp.Pool(processes=len(groups))
    mgr = mp.Manager()
    ns = mgr.Namespace()
    ns.df = df 
    args = [(ns, group) for group in groups]
    result = pool.starmap(process, args)
    pool.close()
    pool.join()
    result = [df] + result
    return pd.concat(result, axis=1)


def make_td(df, cols):
    """ make first flag for given columns
    """
    print("loaded")
    if isinstance(cols, list):
        c = '_'.join(cols)
    else:
        c = cols
    fc_col = f"{c}_ft"
    td_col = f"{c}_td2"
    # df = df.reset_index().sort_values('click_time')
    print("sorted")
    first_click = df.groupby(cols).click_time.shift(-1)
    print("shifted")
    df[fc_col] = first_click
    df[td_col] = df[fc_col] - df.click_time
    print("delta calculated")
    df = df.drop([fc_col], axis=1)
    vx = df[df[td_col].notnull()].groupby(cols)[td_col].median()
    vx = vx.reset_index().rename(columns={td_col: 'med'})
    df = df.merge(vx, on=cols, how='left')
    df.loc[df[td_col].isnull(), td_col] = df.loc[df[td_col].isnull(), 'med']
    print("median imputing end")
    df[td_col] = df[td_col].fillna(-1)
    df.drop('med', axis=1, inplace=True)
    # df = df.set_index('index').sort_index()
    print("end")
    return df[td_col].to_frame(), td_col


def make_diff(df, cols):
    """ make time delta for given columns
    """
    ct = "click_time"
    if isinstance(cols, list):
        td_col = f"{'_'.join(cols)}_delta"
    else:
        td_col = f"{cols}_delta"
    df = df.reset_index().sort_values(ct)
    df[td_col] = df.groupby(cols)[[ct]].transform(lambda x: x.diff())
    td = df.set_index('index').sort_index()
    td = td.fillna(10**10).reset_index().rename(columns={'click_time': td_col})
    return td[td_col].to_frame(), td_col


def make_diff2(df, group):
    col_name = '_'.join(group)
    new_feature = col_name + '_nextClick_test'
    D = 2**26
    # df = df.sort_values('click_time')
    # df = df.reset_index()
    df['category'] = ''
    for c in tqdm(group):
        df['category'] += df[c].astype(str) + "_"
    df['category'] = df['category'].apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    next_clicks = []
    for category, t in tqdm(zip(reversed(df['category'].values), reversed(df['click_time'].values))):
        next_clicks.append(click_buffer[category]-t)
        click_buffer[category] = t
    del(click_buffer)
    QQ = list(reversed(next_clicks))
    df[new_feature] = pd.Series(QQ).astype('float32')
    gc.collect()
    # df = df.set_index('index').sort_index()
    return df[new_feature].to_frame(), new_feature


def make_diff3(df, group):
    col_name = '_'.join(group)
    new_feature = col_name + '_nextClick2'
    D = 2**26
    df = df.reset_index().sort_values('click_time')
    df['category'] = ''
    for c in tqdm(group):
        df['category'] += df[c].astype(str) + "_"
    df['category'] = df['category'].apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    next_clicks = []
    for category, t in tqdm(zip(reversed(df['category'].values), reversed(df['click_time'].values))):
        next_clicks.append(click_buffer[category]-t)
        click_buffer[category] = t
    del(click_buffer)
    QQ = list(reversed(next_clicks))
    df[new_feature] = pd.Series(QQ).astype('float32')
    vx = df[df[new_feature] <= 10**9].groupby("category")[new_feature].median()
    vx = vx.reset_index().rename(columns={new_feature: 'under_med'})
    df = df.merge(vx, on='category', how='left')
    df.loc[df[new_feature] > 10**9, new_feature] = df.loc[df[new_feature] > 10**9, 'under_med']
    df.drop('under_med', axis=1, inplace=True)
    gc.collect()
    ipdb.set_trace()
    df = df.set_index('index').sort_index()
    df[new_feature] = df[new_feature].fillna(-1)
    ipdb.set_trace()
    return df[new_feature].to_frame(), new_feature


def make_prev(df, group):
    col_name = '_'.join(group)
    new_feature = col_name + '_prevClick'
    D = 2**26
    # df = df.reset_index()
    # df = df.sort_values('click_time')
    df['category'] = ''
    for c in tqdm(group):
        df['category'] += df[c].astype(str) + "_"
    df['category'] = df['category'].apply(hash) % D
    click_buffer = np.full(D, 0, dtype=np.int32)
    next_clicks = []
    click_gen = tqdm(zip(df['category'].values, df['click_time'].values))
    for category, t in click_gen:
        prev_t = click_buffer[category]
        if prev_t == 0:
            next_clicks.append(-1)
        else:
            next_clicks.append(t - prev_t)
        click_buffer[category] = t
    del(click_buffer)
    QQ = list(next_clicks)
    df[new_feature] = pd.Series(QQ).astype('float32')
    gc.collect()
    # df = df.set_index('index').sort_index()
    return df[new_feature].to_frame(), new_feature


def make_next_proc(df, group,
                   target=None,
                   proc=None,
                   shift=1):
    df['next_hour'] = df.hour + 1
    pf_cols = f"{'_'.join(group)}_nextcount"
    group_cols = group + ['hour', 'day']
    vc = df.groupby(group_cols).size().reset_index()
    vc = vc.rename(columns={'hour': 'next_hour', 0: pf_cols})
    merge_cols = group + ['next_hour', 'day']
    df = df.reset_index().merge(vc, on=merge_cols, how='left').set_index('index')
    df = df.sort_index()
    df[pf_cols] = df[pf_cols].fillna(0)
    import ipdb; ipdb.set_trace()
    return df[pf_cols].to_frame(), pf_cols


def make_shift(df, group):
    df, col_name = make_diff2(df, group)
    new_col = col_name + '_shift'
    df[new_col] = df[col_name].shift(+1).values
    return df[new_col].to_frame(), new_col


def make_id(test, sup, group):
    """ merge click_id to sup
    """
    sup = sup.drop('click_id', axis=1).reset_index()
    df = pd.merge(sup, test, on=group, how='left')
    df = df.set_index('index').sort_index()
    df['click_id'] = df.click_id.fillna(-1)
    df.loc[df.duplicated(), 'click_id'] = -1
    cid = df.click_id.to_frame()
    cid.to_csv('preprocessed/features/test_click_id.csv')


def make_sequence(df, cols):
    """ make sequence for given cols
    """
    NotImplemented


def make_ce(df, group):
    """ count encoding for each columns
    """
    if isinstance(group, list):
        ce_col = f"{'_'.join(group)}_ce"
    else:
        ce_col = f"{group}_ce"
    vc = df.groupby(group).size().reset_index()
    vc = vc.rename(columns={0: ce_col})
    df = df.reset_index().merge(vc, on=group).set_index('index')
    df = df.sort_index()
    return df[ce_col].to_frame(), ce_col


def make_proc_to_c(df, group, target, proc):
    """ proc to target column
    """
    if isinstance(group, list):
        ce_col = f"{'_'.join(group)}_{target}_{proc}"
    else:
        ce_col = f"{group}_{target}_{proc}"
    if proc == 'cumcount':
        feat = df.groupby(group)[target].agg(proc).to_frame()
        feat = feat.rename(columns={0: ce_col})
    else:
        vc = df.groupby(group)[target].agg(proc).reset_index()
        vc = vc.rename(columns={target: ce_col})
        df = df.reset_index().merge(vc, on=group).set_index('index')
        df = df.sort_index()
        feat = df[ce_col].to_frame()
    import ipdb; ipdb.set_trace()
    return feat, ce_col
    

def make_comp(df, col_combs, logger, count=False, drop=False):
    
    for combs in col_combs:
        with logger.interval_timer(combs):
            col_name = '_'.join(combs)
            df[col_name] = ""
            for col in combs:
                df[col_name] += df[col].astype(str) + '_'
            if count:
                df = make_ce(df, [col_name], logger)
                if drop:
                    df.drop(col_name, axis=1, inplace=True)
    return df


def make_rolling(df, cols, target, proc, window_size=3600):
    """ rolling mean for each cols
    """
    window_size = int(window_size)
    rm_col = f"{'_'.join(cols)}_rolling{window_size}_{target}_{proc}"
    group_cols = cols + [target]
    df = df.reset_index()
    df.index = df.click_time
    df = df.sort_index()
    vx = df[group_cols].groupby(cols)[target].rolling(window_size).agg(proc)
    cols += ['click_time']
    vx = vx.reset_index().rename(columns={target: rm_col})
    df = df.merge(vx, on=cols).set_index('index').sort_index()
    return df[rm_col].to_frame(), rm_col


def construct_features(proc,
                       group,
                       dump_dir,
                       target=None,
                       subproc=None,
                       test_split=False):
    """ construct feature and dump it distinctly
    """
    df = load_feather(config.TRAIN_PATH,
                      parse_dates=config.TRAIN_PARSE_DATES)
    if test_split:
        test_df = load_feather(config.TEST_PATH,
                               parse_dates=config.TEST_PARSE_DATES)
        train_len = df.shape[0]
        df = pd.concat([df, test_df])
    df = df.reset_index(drop=True)
    df['hour'] = df.click_time.dt.hour
    df['day'] = df.click_time.dt.day
    df['click_time'] = df.click_time.astype('int64') // 10**9
    # df.drop('attributed_time', axis=1, inplace=True)
    if target:
        if subproc:
            f1, col_name = proc(df, group, target, subproc)
        else:
            f1, col_name = proc(df, group, target)
    else:
        f1, col_name = proc(df, group)
    if test_split:
        train_df, test_df = f1[:train_len], f1[train_len:]
        train_df.to_csv(f'{dump_dir}train_{col_name}.csv',
                        index=False)
        test_df.to_csv(f'{dump_dir}test_{col_name}.csv',
                       index=False)
    else:
        f1.to_csv(dump_dir + f'{col_name}.csv',
                  index=False)


def parser():
    parser = ArgumentParser(description='make features')
    parser.add_argument('--cols',
                        nargs='+',
                        help='base group of columns',
                        required=True)
    parser.add_argument('--target',
                        help='target column',
                        required=False)
    parser.add_argument('--subproc',
                        help='proc to target column',
                        required=False)
    return parser
                        

def make_cv(sep_time):
    """
    make cv train/valid csv files in preprocessed folder
    """
    log_name = now()
    log_path = f'log/make_feature/'
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    log_fname = log_path + f'{log_name}.log'
    logger = Logger('make_feature', log_fname)

    df = load_csv(config.TRAIN_PATH,
                  dtypes=config.TRAIN_DTYPES,
                  parse_dates=config.TRAIN_PARSE_DATES)
    df = df.drop('attributed_time', axis=1)
    df['hour'] = df.click_time.dt.hour
    df['day'] = df.click_time.dt.day

    ce_cols = [['ip', 'day', 'hour'],
               ['ip', 'app'],
               ['ip', 'app', 'os'],
               ['ip', 'device'],
               ['app', 'channel']]
    with logger.interval_timer('compute'):
        df = to_parallel(df, make_ce, ce_cols)

    split_gen = enumerate(timeseries_cv(df,
                                        config.SEP_TIME))

    for num, (train_index, test_index) in split_gen:
        logger.info(f"fold {num} start")
        train_df, test_df = df.loc[train_index], df.loc[test_index]
        train_df.to_csv(f'preprocessed/val/train.csv',
                        index=False)
        test_df.to_csv(f'preprocessed/val/test.csv',
                       index=False)


def add_features(dir_name, dump_dir, debug=None):
    """ add features to existed files
    """
    log_name = now()
    log_path = f'log/make_feature/'
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    log_fname = log_path + f'{log_name}.log'
    logger = Logger('make_feature', log_fname)

    Path(dump_dir).mkdir(parents=True, exist_ok=True)

    cv = enumerate(generate_cv(dir_name, 4, debug))
    for num, (train_df, test_df) in cv:
        df = dd.concat([train_df, test_df])
        train_len = len(train_df)
        df['hour'] = df.click_time.dt.hour
        df['day'] = df.click_time.dt.day
        df = make_comp(df,
                       [['ip', 'day', 'hour'],
                        ['ip', 'app'],
                        ['ip', 'app', 'os'],
                        ['ip', 'device'],
                        ['app', 'channel']],
                       logger,
                       count=True)
        df = make_comp(df,
                       [['app', 'channel'],
                        ['app', 'device'],
                        ['app', 'os'],
                        ['os', 'ip'],
                        ['app', 'ip']],
                       logger)

        df = df.compute().reset_index()
        df.drop('index', axis=1, inplace=True)
        
        df = make_diff(df,
                       [['ip', 'app'],
                        ['ip', 'app', 'os'],
                        ['ip', 'device'],
                        ['app', 'channel'],
                        ['ip', 'app', 'os', 'device']],
                       1,
                       logger)
        train_df, test_df = df.loc[:train_len], df.loc[train_len:]
        train_df.to_csv(dump_dir + f'fold_{num}_train.csv',
                        index=False)
        test_df.to_csv(dump_dir + f'fold_{num}_test.csv',
                       index=False)
        

if __name__ == '__main__':
    set_seed(2018)
    dump_dir = 'preprocessed/features/'
    Path(dump_dir).mkdir(exist_ok=True, parents=True)
    parser = parser()
    print(parser.parse_args().cols)
    construct_features(make_diff2,
                       parser.parse_args().cols,
                       dump_dir,
                       # target=parser.parse_args().target,
                       # subproc=parser.parse_args().subproc,
                       test_split=True)
