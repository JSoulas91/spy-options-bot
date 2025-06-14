import json
import matplotlib.pyplot as plt

rewards = []
losses = []
entropies = []
confidences = []

with open("meta/meta_log.jsonl") as f:
    for line in f:
        data = json.loads(line)
        rewards.append(data.get("shaped_reward", 0))
        losses.append(data.get("loss", 0))
        entropies.append(data.get("entropy", 0))
        confidences.append(data.get("avg_confidence", 0))

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