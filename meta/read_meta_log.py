import pandas as pd

# Load your file
df = pd.read_json("meta_log.jsonl", lines=True)

# Add win/loss flag
df["pct_pnl"] = df["pct_pnl"].astype(float)
df["is_win"] = df["pct_pnl"] > 0

# Define bins
bins = [-1000, -50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50, 1000]
labels = [
    "-1000% to -50%", "-50% to -20%", "-20% to -10%", "-10% to -5%",
    "-5% to -2%", "-2% to 0%", "0% to 2%", "2% to 5%", "5% to 10%",
    "10% to 20%", "20% to 50%", "50% to 1000%"
]
df["pnl_bucket"] = pd.cut(df["pct_pnl"], bins=bins, labels=labels, right=False)

# Count wins/losses
summary = df.groupby("pnl_bucket")["is_win"].value_counts().unstack().fillna(0).astype(int)
summary.columns = ["Losses", "Wins"]
summary["Total"] = summary["Losses"] + summary["Wins"]

print(summary)