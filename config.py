TRAIN_DTYPES = {'ip': 'uint32',
                'app': 'uint16',
                'device': 'uint16',
                'os': 'uint16',
                'channel': 'uint16',
                'is_attributed': 'uint8'}
TRAIN_PARSE_DATES = ['click_time', 'attributed_time']
TRAIN_COLS = ['app',
              'device',
              'os',
              'channel',
              'click_hour']
EMBEDD_DIM = {'app': 20,
              'device': 20,
              'os': 20,
              'channel': 20,
              'click_hour': 3}
UN_THR = {'app': 10,
          'device': 10,
          'os': 10,
          'channel': 10,
          'click_hour': 10}
SEP_TIME = ['2017-11-07 16:00:00',
            '2017-11-08 04:00:00',
            '2017-11-08 16:00:00',
            '2017-11-09 04:00:00',
            '2017-11-09 16:00:00']
SAMPLE_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 50
IP_MAX = 126413
TRAIN_PATH = './input/train.csv.zip'
