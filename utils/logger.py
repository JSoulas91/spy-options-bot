import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
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

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Initialize logger
bot_logger = setup_logger('bot_logger', 'logs/bot.log')