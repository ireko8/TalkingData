from pathlib import Path
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.utils import plot_model


class NNManager():
    """setting for Neural Net learning
    """
    def __init__(self,
                 name,
                 version,
                 model,
                 batch_size,
                 train_cols):
        
        self.name = name
        self.version = version
        self.model = model
        self.batch_size = batch_size
        self.train_cols = train_cols

        ver_path = f'model/{name}/{version}/'
        Path(ver_path).mkdir(exist_ok=True)
        
        self.dump_path = ver_path + 'weights.hdf5'
        self.csv_log_path = ver_path + 'epochs.csv'
        
        plot_model(self.model, to_file=ver_path+'arch.png')
        with open(ver_path + 'entity_nn.txt', 'w') as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + '\n'))

        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=10,
                                        verbose=1,
                                        min_delta=0.00001,
                                        mode='min'),
                          ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=4,
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
                          sample_size=None,
                          mode='train'):
        while True:
            if mode == 'train':
                grouped = df.groupby('is_attributed')
                base_df = grouped.apply(lambda x: x.sample(sample_size))
            else:
                base_df = df
            
            base_id = np.random.permutation(len(base_df))
            for start in range(0, len(base_df), self.batch_size):
                end = min(start + self.batch_size, len(base_df))
                batch_df_id = base_id[start:end]
                batch_df = base_df.iloc[batch_df_id]
                x_batch = batch_df[self.train_cols]

                if mode != 'test':
                    y_batch = batch_df.is_attributed.values
                    yield x_batch, y_batch
                    
                else:
                    yield x_batch

    def calc_steps(self, df_size):
        return int(np.ceil(df_size/self.batch_size))
        
    def learn(self, train_df, valid_df, train_cols,
              sample_size,
              validation_steps,
              epochs=20):
        train_gen = self.balanced_data_gen(train_df,
                                           sample_size=sample_size)
        valid_gen = self.balanced_data_gen(valid_df,
                                           mode='valid')
        data_size = sample_size*2  # sample_size * label_num
        valid_steps = self.calc_steps(valid_df.shape[0])
        steps_per_epoch = self.calc_steps(data_size)

        history = self.model.fit_generator(generator=train_gen,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           verbose=1,
                                           callbacks=self.callbacks,
                                           validation_data=valid_gen,
                                           validation_steps=valid_steps)
        return history

    def predict_generator(self, test_df, train_cols):
        steps = self.calc_steps(len(test_df))
        test_gen = self.balanced_data_gen(test_df,
                                          mode='test')
        return self.model.predict_generator(test_gen,
                                            steps)
