# config/logging_config.py
import logging
import logging.config
from settings import Log

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(Log.LOG_FILE),
            'formatter': 'standard',
            'level': 'DEBUG',
            'encoding': 'utf-8'
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}


def setup_logging():
    """
    Configura el sistema de logging según LOGGING dict.
    """
    logging.config.dictConfig(LOGGING)