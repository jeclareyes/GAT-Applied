# utils/timer.py
import time
import logging

logger = logging.getLogger(__name__)


def timeit(func):
    """
    Decorador para medir y loggear el tiempo de ejecución de funciones.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"'{func.__name__}' ejecutado en {elapsed:.2f}s")
        return result

    return wrapper
