from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost.sklearn import XGBClassifier
import preprocess
import config
from model.mlp import embedded_mlp
from utils.manager import NNManager
from utils.utils import set_seed, now, load_csv


def df_to_list(df):
    return [df[c] for c in df.columns]


def experiment():
    df = load_csv(config.TRAIN_PATH)
    print("preprocess")
    df, conf = preprocess.preprocess_for_nn(df)
    print("done")
    print(df.shape[1])
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(df[config.TRAIN_COLS],
                                             df.is_attributed):
        print("start")
        train_df, test_df = df.loc[train_index], df.loc[test_index]
        clf = embedded_mlp(conf)

        nnm = NNManager('Embedded_MLP',
                        now(),
                        clf,
                        32,
                        config.TRAIN_COLS,
                        proc_per_gen=df_to_list)
        
        nnm.learn(train_df, test_df, config.SAMPLE_SIZE)
        xgb = XGBClassifier()
        xgb.fit(train_df[config.TRAIN_COLS],
                train_df.is_attributed)
        pred = xgb.predict(test_df[config.TRAIN_COLS])
        print('xgb score', accuracy_score(test_df.is_attributed,
                                          pred))
        print("end")


if __name__ == '__main__':
    set_seed(2018)
    experiment()
