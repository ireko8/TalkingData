from pathlib import Path
from pprint import pformat
import numpy as np
import dask.dataframe as dd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from preprocess import proc_sparse_nn, proc_all, proc_xgb
from make_features import make_td
import config
from validation import timeseries_cv
from model.mlp import sparse_mlp
from utils.manager import NNManager
from utils.logger import Logger
from utils.utils import set_seed, now, load_csv, df_to_list, dump_module


def experiment():
    """experiment func
    """
    
    cv_name = now()
    cv_log_path = f'cv/Embedded_MLP/{cv_name}/'
    Path(cv_log_path).mkdir(parents=True, exist_ok=True)
    
    log_fname = cv_log_path + 'cv.log'
    cv_logger = Logger('CV_log', log_fname)
    cv_logger.info("Experiment Start")

    # dump configuration
    cv_logger.space()
    cv_logger.info("Configuration:::")
    cv_logger.info(dump_module(config))

    with cv_logger.interval_timer("Data Load"):
        df = load_csv(config.TRAIN_PATH,
                      dtypes=config.TRAIN_DTYPES,
                      parse_dates=config.TRAIN_PARSE_DATES)
        with cv_logger.interval_timer("td"):
            df = make_td(df, ['ip', 'os', 'app', 'device'], cv_logger)
        df = proc_all(df)
        df = df.compute().reset_index()
        # df.drop(['ip'], axis=1)
        cv_logger.info(f"df shape: {df.shape}")
    
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # split_gen = enumerate(skf.split(df[config.TRAIN_COLS], df.is_attributed))
    split_gen = enumerate(timeseries_cv(df,
                                        config.SEP_TIME))

    cv_logger.info("cross validation start")
    cv_logger.double_kiritori()
    aucs = []
    
    for num, (train_index, test_index) in split_gen:
        cv_logger.kiritori()
        cv_logger.info(f"fold {num} start")
        
        fold_log_path = cv_log_path + f'fold_{num}/'
        train_df, test_df = df.loc[train_index], df.loc[test_index]
        train_df.drop('click_time', axis=1, inplace=True)
        test_df.drop('click_time', axis=1, inplace=True)
        # train_df, test_df = proc_xgb(train_df,
        #                              test_df,
        #                              ['ip'],
        #                              cv_logger)

        # with cv_logger.interval_timer("Preprocess"):
        #     train_df, test_df, conf = proc_sparse_nn(train_df,
        #                                              test_df,
        #                                              cv_logger)
            
        # # cv_logger.info("Embedding Configuration")
        # # cv_logger.info(pformat(conf))

        # clf = sparse_mlp(len(conf))

        # cv_logger.info("NN Learn Start")
        # nnm = NNManager(fold_log_path,
        #                 clf,
        #                 config.BATCH_SIZE,
        #                 conf,
        #                 cv_logger)

        # with cv_logger.interval_timer("NN Learn"):
        #     nnm.learn(train_df,
        #               test_df,
        #               config.SAMPLE_SIZE,
        #               epochs=config.EPOCHS)

        # cv_logger.info("NN Learn Done")

        cv_logger.info("XGB Baseline validation")
        xgb = XGBClassifier(n_estimators=400, n_jobs=8)
        conf = list(train_df.columns)
        conf.remove(['click_time', 'is_attributed'])
        eval_set = [(train_df[conf], train_df.is_attributed),
                    (test_df[conf], test_df.is_attributed)]
        xgb.fit(train_df[conf],
                train_df.is_attributed,
                eval_metric="auc",
                eval_set=eval_set,
                early_stopping_rounds=30)
        # pred = xgb.predict_proba(test_df[conf])
        auc = max(xgb.evals_result()['validation_1']['auc'])
        aucs.append(auc)
        
        cv_logger.info(f"naive XGB AUC : {auc}")

        cv_logger.info(f"fold {num} end")

    cv_logger.double_kiritori()
    cv_logger.info("Cross Validation Done")
    cv_logger.info(f"AUC {np.mean(aucs)} +/- {np.std(aucs)}")
    cv_logger.info("Experiment Done")


if __name__ == '__main__':
    set_seed(2018)
    experiment()
