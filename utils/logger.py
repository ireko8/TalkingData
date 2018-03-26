import logging
from contextlib import contextmanager
from datetime import datetime


class Logger:
    """Logging Uitlity Class for monitoring and debugging
    """

    def __init__(self,
                 name,
                 log_fname,
                 log_level=logging.INFO,
                 custom_log_handler=None):

        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        ch = logging.FileHandler(log_fname)
        self.logger.addHandler(ch)
        self.logger.addHandler(logging.StreamHandler())
        
        if custom_log_handler:
            if isinstance(custom_log_handler, list):
                for handler in custom_log_handler:
                    self.logger.addHandler(handler)
            else:
                self.logger.addHandler(handler)

    def kiritori(self):
        self.logger.info('-'*80)

    def double_kiritori(self):
        self.logger.info('='*80)

    def space(self):
        self.logger.info('\n')

    @contextmanager
    def interval_timer(self, name):
        start_time = datetime.now()
        self.logger.info("\n")
        self.logger.info(f"Execution {name} start at {start_time}")
        try:
            yield
        finally:
            end_time = datetime.now()
            td = end_time - start_time
            self.logger.info(f"Execution {name} end at {end_time}")
            self.logger.info(f"Execution Time : {td}")
            self.logger.info("\n")
    
    def __getattr__(self, attr):
        """
        for calling logging class attribute
        if you call attributes of other class, raise AttributeError
        """
        return getattr(self.logger, attr)
