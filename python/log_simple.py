
import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)


setup_logger('record', 'record.log')

logger = logging.getLogger('record')
logger.info('www')
logger.debug('www')
logger.warning('www')
logger.error('www')
logger.critical('www')
print('end')