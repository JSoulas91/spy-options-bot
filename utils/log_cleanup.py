import os
import glob

def cleanup_logs(log_dir='logs', days_old=7):
    if not os.path.exists(log_dir):
        return

    for file_path in glob.glob(f"{log_dir}/*.log"):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")