from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger, Callback
from keras.utils import plot_model


def proc_for_valid(df, train_cols, proc=None):

    if proc:
        valid_X = proc(df[train_cols])
    else:
        valid_X = df[train_cols]
    valid_y = df.is_attributed
    return (valid_X, valid_y)


class AUCLoggingCallback(Callback):
    """Custom logger callback for logging and roc-auc
    """
    def __init__(self, logger, valid_data, test_data):
        super().__init__()
        self.logger = logger
        self.valid_X = valid_data[0]
        self.valid_y = valid_data[1]
        self.test_X = test_data[0]
        self.test_y = test_data[1]

    def on_epoch_end(self, epoch, logs={}):
        acc = logs['acc']
        loss = logs['loss']
        val_loss = logs['val_loss']
        val_acc = logs['val_acc']
        pred = self.model.predict(self.valid_X)
        auc = roc_auc_score(self.valid_y, pred)
        
        mes = f"Epoch {epoch:>2}: loss {loss:.4f}, acc {acc:.4f}, "
        mes += f"val_loss {val_loss:.4f}, val_acc {val_acc:.4f}, "
        mes += f"val_auc {auc:.4f}"
        self.logger.info(mes)

    def on_train_end(self, logs={}):
        pred = self.model.predict(self.test_X)
        test_acc = accuracy_score(self.test_y, pred > 0.5)
        test_auc = roc_auc_score(self.test_y, pred)
        mes = f"Test: acc {test_acc:.4f}, auc {test_auc:.4f}"
        self.logger.info(mes)
    

class NNManager():
    """setting for Neural Net learning
    """
    def __init__(self,
                 dir_path,
                 model,
                 batch_size,
                 train_cols,
                 logger,
                 proc_per_gen=None):

        self.dir_path = dir_path
        self.model = model
        self.batch_size = batch_size
        self.train_cols = train_cols
        self.proc_per_gen = proc_per_gen
        self.logger = logger

        Path(dir_path).mkdir(exist_ok=True, parents=True)
        
        self.dump_path = dir_path + 'weights.hdf5'
        self.csv_log_path = dir_path + 'epochs.csv'
        
        plot_model(self.model, to_file=dir_path+'arch.png')
        with open(dir_path + 'entity_nn.txt', 'w') as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + '\n'))

        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=15,
                                        verbose=1,
                                        min_delta=0.00001,
                                        mode='min'),
                          ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=10,
                                            verbose=1,
                                            epsilon=0.0001,
                                            mode='min'),
                          ModelCheckpoint(monitor='val_loss',
                                          filepath=self.dump_path,
                                          save_best_only=True,
                                          save_weights_only=True,
                                          mode='min'),
                          CSVLogger(self.csv_log_path)]

    def balanced_data_gen(self,
                          df,
                          sample_size=None):
        """balanced data generator for training
        """

        pos_idx = df[df.is_attributed == 1].index.values
        neg_idx = df[df.is_attributed == 0].index.values
        
        while True:
            pos_sample = np.random.choice(pos_idx, sample_size)
            neg_sample = np.random.choice(neg_idx, sample_size)
            sample_idx = np.concatenate([pos_sample, neg_sample])
            base_df = df.loc[sample_idx]
            base_id = np.random.permutation(len(base_df))

            pos_size = len(base_df[base_df.is_attributed == 1])
            neg_size = len(base_df[base_df.is_attributed == 0])
            assert(pos_size == neg_size)

            for start in range(0, len(base_df), self.batch_size):
                end = min(start + self.batch_size, len(base_df))
                batch_df_id = base_id[start:end]
                batch_df = base_df.iloc[batch_df_id]
                x_batch = batch_df[self.train_cols]

                if self.proc_per_gen:
                    x_batch = self.proc_per_gen(x_batch)

                y_batch = batch_df.is_attributed.values
                yield x_batch, y_batch

    def calc_steps(self, df_size):
        return int(np.ceil(df_size/self.batch_size))
        
    def learn(self, train_df, valid_df,
              sample_size,
              valid_sampling=None,
              epochs=20):
        """main func fot NN training
        """

        # generator for training
        train_gen = self.balanced_data_gen(train_df,
                                           sample_size=sample_size)
        data_size = sample_size*2  # sample_size * label_num
        steps_per_epoch = self.calc_steps(data_size)

        # validation data
        valid_grouped = valid_df.groupby('is_attributed')
        valid_samples = valid_grouped.apply(lambda x: x.sample(valid_sampling))
        valid_data = proc_for_valid(valid_samples,
                                    self.train_cols,
                                    self.proc_per_gen)
        
        # test data for validation when train end
        test_data = proc_for_valid(valid_df,
                                   self.train_cols,
                                   self.proc_per_gen)

        # to calculate auc score on each end of epoch
        auc_callback = AUCLoggingCallback(self.logger,
                                          valid_data,
                                          test_data)
        self.callbacks.append(auc_callback)

        history = self.model.fit_generator(generator=train_gen,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           verbose=1,
                                           callbacks=self.callbacks,
                                           validation_data=valid_data)
        return history

    def predict_generator(self, test_df):
        test_X = self.proc_per_gen(test_df[self.train_cols])
        return self.model.predict(test_X)
