import logging
import time
from scheduler import run_scheduler_loop

# Optional: Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("SPY Options Trading Bot starting scheduler loop...")
    try:
        run_scheduler_loop()
    except Exception as e:
        logging.exception(f"Fatal error in main scheduler loop: {e}")
        raise
    finally:
        logging.info("Bot has stopped. Exiting main.py.")

if __name__ == "__main__":
    main()