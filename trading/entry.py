# trading/entry.py
"""
Handles new trade entries (live or simulated) using meta-agent gating.
"""
import random, time, traceback
from datetime import datetime

from utils.logger            import bot_logger
from utils.telegram_utils    import send_telegram_message
from utils.trade_tracker     import TradeTracker
from utils.trade_logger      import log_trade
from utils.metrics_logger    import log_trade_metrics
from utils.vix_utils         import get_current_vix
from meta.meta_state         import build_meta_state_for_entry
from meta.meta_agent         import meta_agent
from strategy                import evaluate_trade_signal
from trade_manager           import execute_trade_with_retries
from config import (
    DEFAULT_POSITION_SIZE, ENABLE_DYNAMIC_SIZING,
    MIN_POSITION_SIZE, MAX_POSITION_SIZE, VIX_MODERATE_THRESHOLD,
    SIMULATION_MODE, DEFAULT_SLIPPAGE_BPS, DEFAULT_SPREAD_BPS,
    SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS
)

trade_tracker = TradeTracker()

# ───────────────────────────────── helpers
def _dynamic_size(conf: float, vix: float) -> float:
    if not ENABLE_DYNAMIC_SIZING:
        return DEFAULT_POSITION_SIZE
    vix_factor = 0.7 if vix >= VIX_MODERATE_THRESHOLD else 1.0
    raw = DEFAULT_POSITION_SIZE * conf * vix_factor
    return max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, raw))

def _simulate_fill(mkt: float, side: str):
    spread = mkt * DEFAULT_SPREAD_BPS / 10_000
    mid    = mkt + (spread/2 if side == "buy" else -spread/2)
    slip   = mkt * DEFAULT_SLIPPAGE_BPS / 10_000
    fill   = mid + (slip if side == "buy" else -slip)
    delay  = random.randint(SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS)
    time.sleep(delay/1000)
    return round(fill, 4), delay

# ───────────────────────────────── core
def handle_entry(market_data: dict):
    try:
        now = datetime.now()
        if now.hour == 15 and now.minute >= 30:
            return

        signal = evaluate_trade_signal(market_data)
        if not signal.get("should_trade"):
            return

        confidence  = signal["confidence"]
        trade_type  = signal["trade_type"]
        trade_setup = signal["trade_setup"]

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

        meta_agent.eval_mode()
        meta_action = meta_agent.select_action(meta_state)
        meta_params = meta_agent.interpret_action(meta_action)

        agent_conf = meta_params.get("agent_confidence", 0)

        bot_logger.info(f"[MetaAgent Entry] action={meta_action} agent_conf={agent_conf:.3f}")
        if meta_action == 0 or agent_conf < 0.5:
            bot_logger.info("[MetaAgent Entry] Blocked by agent.")
            send_telegram_message("🛑 Entry blocked by meta-agent (action=0 or low confidence).")
            return

        if not trade_tracker.can_place_trade():
            return

        # --------------- live vs sim ---------------
        if SIMULATION_MODE:
            m_price = market_data["price"]
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
            t0 = time.time()
            order  = execute_trade_with_retries(trade_setup)
            if not order:
                return
            latency  = int((time.time()-t0)*1000)
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
            "option_symbol": trade_setup.get("option_symbol"),
            "meta_action": int(meta_action),
            "meta_confidence": float(agent_conf)
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
            f"fill {order['fill_price']}  slip {slippage*100:.2f}%"
        )

    except Exception as exc:
        bot_logger.error(f"[Entry] {exc}")
        bot_logger.debug(traceback.format_exc())
        send_telegram_message(f"⚠️ Entry error\n{exc}")