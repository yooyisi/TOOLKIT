# -*- coding: utf-8 -*-
import logging.config

conf_dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'DEBUG',  # 打日志的等级可以换的，下面的同理
            'handlers': ['error_file', 'console_handler'],  # 对应下面的键
            'propagate': 1
        },

        'gunicorn.access': {
            'level': 'DEBUG',
            'handlers': ['access_file', 'console_handler'],
            'propagate': 0,
            'qualname': 'access_log'
        }
    },
    'handlers': {
        'console_handler': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'generic',
            'stream': 'ext://sys.stdout'
        },
        'error_file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': 'INFO',
            'when': 'D',
            'backupCount': 10,  # 备份多少份，经过测试，最少也要写1，不然控制不住大小
            'formatter': 'generic',  # 对应下面的键
            # 'mode': 'w+',
            'filename': 'logs/error.log',  # 打日志的路径
            'encoding': 'utf-8'
        },
        'access_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 1024 * 1024 * 1024,
            'backupCount': 1,
            'formatter': 'generic',
            'filename': 'logs/access.log',
        }
    },
    'formatters': {
        'generic': {
            'format': '[%(process)d] [%(asctime)s] %(levelname)s [%(pathname)s:%(lineno)s] %(message)s',  # 打日志的格式
            'datefmt': '[%Y-%m-%d %H:%M:%S %z]',  # 时间显示方法
            'class': 'logging.Formatter'
        },
        'access': {
            'format': '[%(process)d] [%(asctime)s] %(levelname)s [%(pathname)s:%(lineno)s] %(message)s',
            'class': 'logging.Formatter'
        }
    }
}

logging.config.dictConfig(conf_dict)
logger = logging.getLogger(__name__)
logger.info('...')
