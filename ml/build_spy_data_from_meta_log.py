#ml/build_spy_data_from_meta_log.py

import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from technical_analysis.indicators import calculate_indicators

# === Config ===
LOG_PATH = os.path.join("meta", "meta_log.jsonl")
OUTPUT_PATH = os.path.join("ml", "spy_data.csv")

def extract_rows(log_path):
    rows = []
    with open(log_path, "r") as f:
        for line in tqdm(f, desc="Parsing log"):
            try:
                obj = json.loads(line.strip())
                ts = obj.get("timestamp")
                state = obj.get("state", {})
                trade = obj.get("trade", {})
                if not ts or not state:
                    continue

                row = {
                    "timestamp": ts,
                    "open": state.get("open"),
                    "high": state.get("high"),
                    "low": state.get("low"),
                    "close": state.get("close"),
                    "volume": state.get("volume"),
                }

                # === Add label from meta_action (or reward)
                # You can swap this line to: label = obj.get("reward") if preferred
                label = trade.get("meta_action")

                if label is not None:
                    row["label"] = label
                else:
                    continue  # skip rows with no label

                if all(v is not None for v in row.values()):
                    rows.append(row)

            except Exception as e:
                continue  # skip malformed lines

    return pd.DataFrame(rows)

def main():
    if not os.path.exists(LOG_PATH):
        print(f"[Error] Cannot find log file: {LOG_PATH}")
        return

    df = extract_rows(LOG_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[Info] Extracted {len(df)} labeled rows from meta_log.jsonl")

    if len(df) < 50:
        print("[Warning] Less than 50 rows — retraining may fail.")

    df = calculate_indicators(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[Saved] spy_data.csv written to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()