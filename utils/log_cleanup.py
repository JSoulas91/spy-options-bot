import os
import glob
import time
from datetime import datetime
from utils.logger import bot_logger

def delete_old_files(folder: str, pattern: str, max_age_days: int):
    now = time.time()
    deleted_files = 0
    for file_path in glob.glob(os.path.join(folder, pattern)):
        try:
            if not os.path.isfile(file_path):
                continue

            file_age_days = (now - os.path.getmtime(file_path)) / 86400
            if file_age_days > max_age_days:
                os.remove(file_path)
                bot_logger.info(f"🧹 Deleted old file: {file_path} ({file_age_days:.1f} days old)")
                deleted_files += 1
        except Exception as e:
            bot_logger.error(f"⚠️ Error deleting {file_path}: {e}")

    return deleted_files

def cleanup_logs_and_backups(log_dir='logs', backup_dir='backups'):
    bot_logger.info("🧹 Starting log and backup cleanup process...")

    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    # Cleanup logs
    log_deleted = delete_old_files(log_dir, "*.log", max_age_days=7)
    bak_deleted = delete_old_files(log_dir, "*.bak", max_age_days=1)
    csv_deleted = delete_old_files(log_dir, "*.csv", max_age_days=90)

    # Cleanup backups
    backup_deleted = delete_old_files(backup_dir, "*", max_age_days=7)

    total_deleted = log_deleted + bak_deleted + csv_deleted + backup_deleted

    if total_deleted == 0:
        bot_logger.info("✅ No old files found for deletion.")
    else:
        bot_logger.info(f"🗑️ Cleanup complete. {total_deleted} old files deleted.")

# Only for standalone testing
if __name__ == "__main__":
    cleanup_logs_and_backups()