import logging
from logging import Logger
from typing import Optional
# the app level logger
def get_logger(name:str, level:int = logging.INFO, formatter: Optional[logging.Formatter] = None) -> Logger:
    """
    Return a logger instance with a given name and logging level.
    If no formatter is provided, a default formatter will be used.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # log to console
        """
        ch = logging.StreamHandler()
        ch.setLevel(level)
        # Use a default formatter if none is provided
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        ch.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(ch)
        """
        # log to file
        if formatter is None:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
        file_handler = logging.FileHandler(f"{name}.log", mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

# get global logger if there's no app-level loggger
global_logger:Logger = None
def get_global_logger()->logging.Logger:
    global global_logger
    if global_logger is None:
         global_logger = get_logger("openagents", level=logging.DEBUG)
    return global_logger
