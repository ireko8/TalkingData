from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger, Callback
from keras.utils import plot_model


class AUCLoggingCallback(Callback):
    """Custom logger callback for logging and roc-auc
    """
    def __init__(self, logger, valid_X, valid_y):
        super().__init__()
        self.logger = logger
        self.valid_X = valid_X
        self.valid_y = valid_y

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
        while True:
            grouped = df.groupby('is_attributed')
            base_df = grouped.apply(lambda x: x.sample(sample_size))
            base_id = np.random.permutation(len(base_df))
            # print("ip count", len(base_df.ip.unique()))

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
              epochs=20):
        """main func fot NN training
        """

        # generator for training
        train_gen = self.balanced_data_gen(train_df,
                                           sample_size=sample_size)
        data_size = sample_size*2  # sample_size * label_num
        steps_per_epoch = self.calc_steps(data_size)

        # validation data
        valid_X = self.proc_per_gen(valid_df[self.train_cols])
        valid_y = valid_df.is_attributed
        valid_data = (valid_X, valid_y)

        # to calculate auc score on each end of epoch
        auc_callback = AUCLoggingCallback(self.logger,
                                          valid_X,
                                          valid_y)
        self.callbacks.append(auc_callback)

        history = self.model.fit_generator(generator=train_gen,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           verbose=0,
                                           callbacks=self.callbacks,
                                           validation_data=valid_data)
        return history

    def predict_generator(self, test_df):
        test_X = self.proc_per_gen(test_df[self.train_cols])
        return self.model.predict(test_X)
