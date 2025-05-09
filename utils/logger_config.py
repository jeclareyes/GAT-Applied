# logger_config.py

import logging

USE_COLORLOG = True  # Puedes controlar esto por config/env

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()

    if USE_COLORLOG:
        try:
            from colorlog import ColoredFormatter
            formatter = ColoredFormatter(
                "%(log_color)s%(levelname)-8s: %(message)s",
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'bold_red',
                }
            )
        except ImportError:
            formatter = logging.Formatter("%(levelname)s: %(message)s")
    else:
        formatter = logging.Formatter("%(levelname)s: %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

     # Suppress noisy DEBUG logs from Fiona, Fiona's GDAL env, and GDAL
    logging.getLogger('fiona').setLevel(logging.WARNING)
    logging.getLogger('fiona._env').setLevel(logging.WARNING)
    logging.getLogger('osgeo').setLevel(logging.WARNING)
