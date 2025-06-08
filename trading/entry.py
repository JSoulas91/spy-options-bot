# trading/entry.py
"""
Handles new trade entries (live or simulated).

➕  Logs every meta‑decision to meta/meta_log.jsonl for online training
➕  Sends Telegram alert if the meta‑agent vetoes a trade
"""

import json, random, time, traceback
from datetime import datetime

from utils.logger            import bot_logger
from utils.telegram_utils    import send_telegram_message
from utils.trade_tracker     import TradeTracker
from utils.trade_logger      import log_trade
from utils.metrics_logger    import log_trade_metrics
from utils.vix_utils         import get_current_vix
from meta.meta_state         import build_meta_state_for_entry
from meta.meta_agent         import MetaAgent
from strategy                import evaluate_trade_signal
from trade_manager           import execute_trade_with_retries
from config import (
    DEFAULT_POSITION_SIZE, ENABLE_DYNAMIC_SIZING,
    MIN_POSITION_SIZE, MAX_POSITION_SIZE, VIX_MODERATE_THRESHOLD,
    SIMULATION_MODE, DEFAULT_SLIPPAGE_BPS, DEFAULT_SPREAD_BPS,
    SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS,
    META_LOG_PATH                                     # <-- new
)

trade_tracker = TradeTracker()
meta_agent    = MetaAgent()       # single instance

# ───────────────────────────────── helpers
def _dynamic_size(conf: float, vix: float) -> float:
    if not ENABLE_DYNAMIC_SIZING:
        return DEFAULT_POSITION_SIZE
    vix_factor = 0.7 if vix >= VIX_MODERATE_THRESHOLD else 1.0
    raw = DEFAULT_POSITION_SIZE * conf * vix_factor
    return max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, raw))

def _simulate_fill(mkt: float, side: str):
    spread = mkt * DEFAULT_SPREAD_BPS / 10_000
    mid    = mkt + (spread / 2 if side == "buy" else -spread / 2)
    slip   = mkt * DEFAULT_SLIPPAGE_BPS / 10_000
    fill   = mid + (slip if side == "buy" else -slip)
    delay  = random.randint(SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS)
    time.sleep(delay / 1000)
    return round(fill, 4), delay

# ───────────────────────────────── core
def handle_entry(market_data: dict):
    try:
        now = datetime.now()
        if now.hour == 15 and now.minute >= 30:
            return

        # 1.  Get raw trading signal from your heuristics
        signal = evaluate_trade_signal(market_data)
        if not signal.get("should_trade"):
            return

        confidence  = signal["confidence"]
        trade_type  = signal["trade_type"]
        trade_setup = signal["trade_setup"]

        # 2.  Build meta‑state & ask the agent
        vix = get_current_vix() or 20
        size = _dynamic_size(confidence, vix)
        trade_setup["size"] = size

        meta_state = build_meta_state_for_entry(
            signal["tf_1m"], signal["tf_5m"], signal["tf_15m"],
            signal["tf_1h"], signal["tf_1d"],
            confidence, trade_type,
            trade_tracker.get_open_trades(),
            signal.get("long_term_data", {}),
            size
        )
        action_idx, agent_conf = meta_agent.select_action(meta_state)
        meta_params = meta_agent.interpret_action(action_idx, agent_conf)

        # ---- log the decision for future online training ----
        try:
            with open(META_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "timestamp":   now.isoformat(),
                    "meta_state":  meta_state.tolist(),
                    "meta_action": action_idx,
                    "agent_conf":  agent_conf,
                    "done": False     # will be updated on exit
                }) + "\n")
        except Exception as e:
            bot_logger.warning(f"[Meta‑log] {e}")

        # 3.  If agent vetoes → skip + alert
        if action_idx == 0:
            send_telegram_message(
                f"⛔️ Meta‑agent vetoed trade (conf {agent_conf:.2f})."
            )
            return

        # 4.  Risk gate: max open trades
        if not trade_tracker.can_place_trade():
            return

        # --------------- live vs sim ---------------
        if SIMULATION_MODE:
            m_price  = market_data["price"]
            fill_px, latency = _simulate_fill(m_price, "buy")
            order = {
                "id": f"sim_{int(time.time()*1000)}",
                "symbol": trade_setup.get("symbol", "SPY"),
                "fill_price": fill_px,
                "status": "filled",
                "latency_ms": latency,
            }
            slippage = abs(fill_px - m_price) / m_price
        else:
            t0     = time.time()
            order  = execute_trade_with_retries(trade_setup)
            if not order:
                return
            latency  = int((time.time() - t0) * 1000)
            slippage = abs(order.get("fill_price", market_data["price"]) -
                           market_data["price"]) / market_data["price"]

        # ---- bookkeeping
        log_trade_metrics(order["id"], latency, slippage)
        order.update({
            "confidence": confidence,
            "size": size,
            "timestamp": now.isoformat(),
            "trade_type": trade_type,
            "meta_state": meta_state.tolist(),
            "meta_action": action_idx,
            "agent_conf": agent_conf,
            "option_symbol": trade_setup.get("option_symbol")
        })
        trade_tracker.log_trade(order, trade_type)
        log_trade({
            "timestamp": now.isoformat(),
            "action": "buy",
            "symbol": order["symbol"],
            "size": size,
            "confidence": confidence,
            "trade_id": order["id"],
        })

        send_telegram_message(
            f"🟢 Enter {order['symbol']} size {size:.2%} "
            f"fill {order['fill_price']}  slip {slippage*100:.2f}% "
            f"(agent conf {agent_conf:.2f})"
        )

    except Exception as exc:
        bot_logger.error(f"[Entry] {exc}")
        bot_logger.debug(traceback.format_exc())
        send_telegram_message(f"⚠️ Entry error\n{exc}")