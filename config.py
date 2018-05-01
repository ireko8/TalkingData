TRAIN_DTYPES = {'ip': 'uint32',
                'app': 'uint16',
                'device': 'uint16',
                'os': 'uint16',
                'channel': 'uint16',
                'is_attributed': 'uint8'}
TRAIN_PARSE_DATES = ['click_time']
TEST_DTYPES = {'click_id': 'int32',
               'ip': 'uint32',
               'app': 'uint16',
               'device': 'uint16',
               'os': 'uint16',
               'channel': 'uint16'}
TEST_PARSE_DATES = ['click_time']

TRAIN_COLS = ['ip_te',
              'app_channel_te',
              'device_os_te',
              'app_os_te',
              'nextClick',
              'ip_day_hour_channel_count',
              'ip_app_nunique']
ENCODE_LIST = [['ip']]
    # ['ip', 'app'],
    # ['app', 'channel']]
    # ['channel', 'os'],
    # ['app', 'os']]
TE_THR = 1000

EMBEDD_DIM = {'app': 20,
              'device': 20,
              'os': 20,
              'channel': 20,
              'click_hour': 3}
UN_THR = {'app': 100,
          'device': 100,
          'os': 100,
          'channel': 100,
          'click_hour': 100}
SEP_TIME = ['2017-11-08 16:00:00',
            '2017-11-09 16:00:00']
SAMPLE_SIZE = 20000
VALID_SIZE = 20000
BATCH_SIZE = 1024
EPOCHS = 50
IP_MAX = 126413
TRAIN_PATH = 'input/train.csv'
TEST_PATH = 'input/test_supplement.csv'
PREPROCESSED_FEATURES = 'preprocessed/all_td.csv'
PREPROCESSED_TEST = 'preprocessed/test.csv'
