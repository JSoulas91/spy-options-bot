import json
from meta.reward_shaper import compute_reward

def load_trades(filename, max_trades=100):
    trades = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i >= max_trades:
                break
            try:
                trade = json.loads(line)
                trades.append(trade)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {i+1}")
    return trades

def analyze_trades(trades):
    rewards = []
    print(f"Analyzing {len(trades)} trades...\n")
    for i, trade in enumerate(trades):
        # Market info for reward calculation
        market = {
            "realized_vol": trade.get("realized_vol", 1.0),
            "vix": trade.get("vix", 15.0),
        }
        reward = compute_reward(trade, market, "Meta-agent signal")
        rewards.append(reward)

        # Safely format pnl
        pnl = trade.get('pnl')
        pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int, float)) else "NA"

        # For other fields, just get or default "NA"
        confidence = trade.get('confidence', 'NA')
        setup_quality = trade.get('setup_quality', 'NA')

        print(f"Trade {i+1}: pnl={pnl_str}, confidence={confidence}, setup_quality={setup_quality}, reward={reward:.4f}")

    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"\nAverage reward: {avg_reward:.4f}")
    print(f"Min reward: {min(rewards):.4f}")
    print(f"Max reward: {max(rewards):.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_sim_trades.py <meta_log.jsonl> [max_trades]")
        sys.exit(1)

    filename = sys.argv[1]
    max_trades = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    trades = load_trades(filename, max_trades)
    analyze_trades(trades)