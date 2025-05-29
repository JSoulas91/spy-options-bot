import logging
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name: str, log_file: str, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Daily rotating file handler
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",      # Rotate at midnight
        interval=1,           # Every 1 day
        backupCount=7         # Keep last 7 log files
    )
    file_handler.suffix = "%Y-%m-%d"  # Log files will be like bot.log.2025-05-28
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if re-imported
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Initialize rotating logger
bot_logger = setup_logger('bot_logger', 'logs/bot.log')