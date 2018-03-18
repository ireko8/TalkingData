UN_THR = 5

TRAIN_COLS = ['ip',
              'app',
              'device',
              'os',
              'channel',
              'click_hour']
EMBEDD_DIM = {'ip': 100,
              'app': 20,
              'device': 20,
              'os': 15,
              'channel': 20,
              'click_hour': 3}
SAMPLE_SIZE = 10000
TRAIN_PATH = './input/debug.csv'
