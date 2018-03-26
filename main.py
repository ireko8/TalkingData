from pathlib import Path
from pprint import pformat
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from preprocess import proc_emb_nn, proc_all
import config
from model.mlp import embedded_mlp
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
        df = load_csv(config.TRAIN_PATH)
        df = proc_all(df)
        cv_logger.info(f"df shape: {df.shape}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    split_gen = enumerate(skf.split(df[config.TRAIN_COLS], df.is_attributed))

    cv_logger.info("cross validation start")
    cv_logger.double_kiritori()
    
    for num, (train_index, test_index) in split_gen:
        cv_logger.kiritori()
        cv_logger.info(f"fold {num} start")
        
        fold_log_path = cv_log_path + f'fold_{num}/'
        train_df, test_df = df.loc[train_index], df.loc[test_index]

        with cv_logger.interval_timer("Preprocess"):
            train_df, test_df, conf = proc_emb_nn(train_df,
                                                  test_df,
                                                  cv_logger)

        cv_logger.info("Embedding Configuration")
        cv_logger.info(pformat(conf))

        clf = embedded_mlp(conf)

        cv_logger.info("NN Learn Start")
        nnm = NNManager(fold_log_path,
                        clf,
                        config.BATCH_SIZE,
                        config.TRAIN_COLS,
                        cv_logger,
                        proc_per_gen=df_to_list)

        with cv_logger.interval_timer("NN Learn"):
            nnm.learn(train_df,
                      test_df,
                      config.SAMPLE_SIZE,
                      epochs=config.EPOCHS)

        cv_logger.info("NN Learn Done")

        cv_logger.info("XGB Baseline validation")
        xgb = XGBClassifier()
        xgb.fit(train_df[config.TRAIN_COLS],
                train_df.is_attributed)
        pred = xgb.predict_proba(test_df[config.TRAIN_COLS])
        auc = roc_auc_score(test_df.is_attributed, pred[:, 1])
        
        cv_logger.info(f"naive XGB AUC : {auc}")

        cv_logger.info(f"fold {num} end")

    cv_logger.double_kiritori()
    cv_logger.info("Cross Validation Done")
    cv_logger.info("Experiment Done")


if __name__ == '__main__':
    set_seed(2018)
    experiment()
