import json
import pandas as pd
import numpy as np
from pathlib import Path

META_LOG = Path("meta/meta_log.jsonl")
CSV_PATH = Path("ml/spy_data.csv")
NPZ_PATH = Path("ml/dataset.npz")
EXPECTED_STATE_DIM = 83
EXPECTED_FEATURE_COUNT = 21  # matches your classifier feature count

def check_meta_log():
    print("🔍 Checking meta_log.jsonl...")

    if not META_LOG.exists():
        raise FileNotFoundError("🚨 meta_log.jsonl not found")

    malformed = 0
    total = 0

    with META_LOG.open("r") as f:
        for line in f:
            total += 1
            try:
                entry = json.loads(line)
                state = entry.get("meta_state", [])
                if len(state) != EXPECTED_STATE_DIM:
                    malformed += 1
            except json.JSONDecodeError:
                malformed += 1

    print(f"✅ Checked {total} entries.")
    if malformed > 0:
        raise ValueError(f"🚨 Found {malformed} malformed meta_state entries")
    print("✅ All meta_state entries are valid.")

def check_csv():
    print("🔍 Checking spy_data.csv...")

    if not CSV_PATH.exists():
        raise FileNotFoundError("🚨 spy_data.csv not found")

    df = pd.read_csv(CSV_PATH)
    actual_features = [c for c in df.columns if c not in ("timestamp", "label")]

    if len(actual_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"🚨 spy_data.csv has {len(actual_features)} features, expected {EXPECTED_FEATURE_COUNT}")

    if df.isna().any().any():
        raise ValueError("🚨 Found NaN values in spy_data.csv")

    print(f"✅ spy_data.csv contains {len(df)} rows and {len(actual_features)} features")
    print("✅ No missing values.")

def check_npz():
    print("🔍 Checking dataset.npz...")

    if not NPZ_PATH.exists():
        raise FileNotFoundError("🚨 dataset.npz not found")

    data = np.load(NPZ_PATH)
    X, y, ts = data["X"], data["y"], data["timestamps"]

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"🚨 dataset.npz has feature dim {X.shape[1]}, expected {EXPECTED_FEATURE_COUNT}")

    if len(X) != len(y) or len(y) != len(ts):
        raise ValueError("🚨 Mismatch in lengths of X, y, timestamps")

    print(f"✅ dataset.npz: {X.shape[0]} samples, {X.shape[1]} features")

def run_all_checks():
    try:
        check_meta_log()
        check_csv()
        check_npz()
        print("\n🎉 All pipeline checks passed successfully!")
    except Exception as e:
        print(str(e))
        print("❌ Pipeline validation failed.")

if __name__ == "__main__":
    run_all_checks()