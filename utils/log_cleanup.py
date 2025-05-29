import os
import glob
import time
import traceback
from datetime import datetime
from utils.logger import bot_logger

def delete_old_files(folder: str, pattern: str, max_age_days: int) -> int:
    """
    Delete files matching pattern in folder if older than max_age_days.
    """
    now = time.time()
    deleted_files = 0

    try:
        full_pattern = os.path.join(folder, pattern)
        bot_logger.debug(f"🔍 Scanning: {full_pattern}")

        for file_path in glob.glob(full_pattern):
            try:
                if not os.path.isfile(file_path):
                    continue

                file_age_days = (now - os.path.getmtime(file_path)) / 86400
                if file_age_days > max_age_days:
                    os.remove(file_path)
                    bot_logger.info(f"🧹 Deleted: {file_path} ({file_age_days:.1f} days old)")
                    deleted_files += 1
            except Exception as e:
                bot_logger.error(f"⚠️ Error deleting file: {file_path} | {e}")
                bot_logger.debug(traceback.format_exc())

    except Exception as e:
        bot_logger.error(f"❌ Failed to scan {folder} with pattern {pattern} | {e}")
        bot_logger.debug(traceback.format_exc())

    return deleted_files


def cleanup_logs_and_backups(log_dir='logs', backup_dir='backups'):
    """
    Delete old logs and backups with specific retention policies.
    """
    bot_logger.info("🧹 Starting log and backup cleanup process...")

    try:
        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)

        bot_logger.debug(f"📂 Ensured directories: {log_dir}, {backup_dir}")

        # Logs cleanup
        log_deleted = delete_old_files(log_dir, "*.log", max_age_days=7)
        bak_deleted = delete_old_files(log_dir, "*.bak", max_age_days=1)
        csv_deleted = delete_old_files(log_dir, "*.csv", max_age_days=90)

        # Backups cleanup
        backup_deleted = delete_old_files(backup_dir, "*", max_age_days=7)

        total_deleted = log_deleted + bak_deleted + csv_deleted + backup_deleted

        if total_deleted == 0:
            bot_logger.info("✅ No old files found for deletion.")
        else:
            bot_logger.info(
                f"🗑️ Cleanup summary: "
                f"{log_deleted} .log, {bak_deleted} .bak, {csv_deleted} .csv, "
                f"{backup_deleted} backups deleted."
            )
    except Exception as e:
        bot_logger.error(f"❌ Cleanup process failed: {e}")
        bot_logger.debug(traceback.format_exc())

# For manual testing
if __name__ == "__main__":
    cleanup_logs_and_backups()