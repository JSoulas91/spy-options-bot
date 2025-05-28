import shutil
import datetime
import os

def backup_data(source_dir='data', backup_dir='backups'):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    backup_path = os.path.join(backup_dir, f"backup_{today}")

    try:
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        if os.path.exists(source_dir):
            shutil.copytree(source_dir, backup_path)
    except Exception as e:
        print(f"Backup error: {e}")