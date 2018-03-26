TRAIN_COLS = ['app',
              'device',
              'os',
              'channel',
              'click_hour']
EMBEDD_DIM = {'app': 50,
              'device': 50,
              'os': 50,
              'channel': 50,
              'click_hour': 3}
UN_THR = {'app': 10,
          'device': 10,
          'os': 10,
          'channel': 10,
          'click_hour': 10}
SAMPLE_SIZE = 75000
BATCH_SIZE = 1024
EPOCHS = 50
TRAIN_PATH = './input/debug.csv'
