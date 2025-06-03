import logging
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name: str, log_file: str, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s — %(levelname)s — %(module)s — %(funcName)s — Line %(lineno)d — %(message)s'
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=7
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Initialize rotating logger
bot_logger = setup_logger('bot_logger', 'logs/bot.log')