import json, time, os
from utils.logger import bot_logger as logger

_METRICS_PATH = "logs/trade_metrics.jsonl"

def log_trade_metrics(trade_id: str, latency_ms: int, slippage: float):
    rec = {
        "ts": int(time.time()*1000),
        "trade_id": trade_id,
        "latency_ms": latency_ms,
        "slippage": slippage,
    }
    try:
        with open(_METRICS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        logger.warning(f"[Metrics] failed to write: {e}")