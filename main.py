import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import plot_model
import preprocess
from model.mlp import MLP
from utils.load import load_csv
from utils.manager import NNManager
from utils.utils import set_seed, now


def experiment():
    df = load_csv('./input/talkingdata-adtracking-fraud-detection/train_sample.csv.zip')
    print("preprocess")
    df_X, df_y, _, conf = preprocess.preprocess_for_nn(df)
    print("done")
    print(df_X.shape[1])
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = list()
    for train_index, test_index in skf.split(df_X, df_y):
        print("start")
        print(df_X.shape)
        train_X, train_y = df_X.loc[train_index], df_y[train_index]
        test_X, test_y = df_X.loc[test_index], df_y[test_index]
        train_nn_X = [train_X[c] for c in train_X.columns]
        test_nn_X = [test_X[c] for c in test_X.columns]
        clf = MLP(df_X.shape[1], embedded=conf)
        plot_model(clf.model)
        with open('entity_nn.txt', 'w') as fp:
            clf.model.summary(print_fn=lambda x: fp.write(x + '\n'))
        clf.fit(train_nn_X, train_y, epochs=5)
        predict = clf.predict(test_nn_X)
        print(roc_auc_score(test_y, predict))
        scores.append(roc_auc_score(test_y, predict))
        print("end")

    scores = np.array(scores)
    print(scores.mean(), scores.std())


if __name__ == '__main__':
    set_seed(2018)
    experiment()
