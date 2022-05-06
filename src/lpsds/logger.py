import logging
import logging.handlers
import os
#import warnings

#class Singleton(type):
#    _instances = {}
#    def __call__(cls, *args, **kwargs):
#        if cls not in cls._instances:
#            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#        return cls._instances[cls]

#class Logger(metaclass=Singleton):
class Logger():
    def __init__(self):
        level = os.environ.get('LPSDS_LOG_LEVEL', 'DEBUG')
        svc_name = os.environ.get('LPSDS_LOG_SVC_NAME', 'job')
        logLevel = getattr(logging, level)
#        warnings.filterwarnings('default')
#        logging.captureWarnings(True)
        self.logger = logging.getLogger(svc_name)
        self.logger.setLevel(logLevel)

        stderr_log_handler = logging.StreamHandler()
        stderr_log_handler.setLevel(logLevel)
        stderr_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(module)s - %(levelname)s: %(message)s'))
        self.logger.addHandler(stderr_log_handler)


log = Logger().logger
