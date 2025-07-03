import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# File path (adjust if needed)
META_LOG_PATH = "meta/meta_log.jsonl"

# Storage
rewards, losses, entropies, confidences = [], [], [], []
entry_pad_indices = []
exit_pad_indices = []
entry_lengths = []
exit_lengths = []

def is_padded(state):
    return isinstance(state, list) and all(x == 0.5 for x in state)

# Scan log
with open(META_LOG_PATH) as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)

            # Collect metrics
            rewards.append(data.get("shaped_reward", 0))
            losses.append(data.get("loss", 0))
            entropies.append(data.get("entropy", 0))
            confidences.append(data.get("avg_confidence", 0))

            # Padding detection
            entry = data.get("meta_entry_state", [])
            exit_ = data.get("meta_exit_state", [])

            if is_padded(entry):
                entry_pad_indices.append(i)
            if is_padded(exit_):
                exit_pad_indices.append(i)

            entry_lengths.append(len(entry))
            exit_lengths.append(len(exit_))

        except Exception as e:
            print(f"⚠️ Failed to parse line {i}: {e}")

# --- Diagnostics Summary ---
print("\n🔍 Meta Log Diagnostics")
print("=" * 30)
print(f"Total entries: {i+1}")
print(f"Padded meta_entry_state: {len(entry_pad_indices)}")
print(f"Padded meta_exit_state:  {len(exit_pad_indices)}")

if entry_pad_indices:
    print(f"First padded entry at index: {entry_pad_indices[0]}")
if exit_pad_indices:
    print(f"First padded exit at index:  {exit_pad_indices[0]}")

entry_len_counts = Counter(entry_lengths)
exit_len_counts = Counter(exit_lengths)

print("\nMeta Entry Length Distribution:")
for k, v in sorted(entry_len_counts.items()):
    print(f"  Length {k}: {v} entries")

print("\nMeta Exit Length Distribution:")
for k, v in sorted(exit_len_counts.items()):
    print(f"  Length {k}: {v} entries")

# --- Plots ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(rewards, label="Shaped Reward")
plt.title("Reward Curve")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(losses, label="Loss", color="orange")
plt.title("Loss Trend")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(confidences, label="Avg Confidence", color="green")
plt.title("Confidence Behavior")
plt.grid(True)

plt.tight_layout()
plt.show()