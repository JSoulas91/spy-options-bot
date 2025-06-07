# trading/entry.py
"""
Handles new trade entries.

• If SIMULATION_MODE == True
      – fills are simulated with slippage + spread
      – a random 50‑150 ms latency is applied
• Otherwise
      – real Tradier order flow is used (as before)
"""

import random, time, sys, traceback
from datetime import datetime
from utils.logger            import bot_logger
from utils.trade_tracker     import TradeTracker
from utils.trade_logger      import log_trade
from utils.metrics_logger    import log_trade_metrics
from utils.telegram_notifier import TelegramNotifier
from utils.vix_utils         import get_current_vix
from meta.meta_agent         import meta_agent
from meta.meta_state         import build_meta_state_for_entry
from strategy                import evaluate_trade_signal
from trade_manager           import execute_trade_with_retries
from config import (
    DEFAULT_POSITION_SIZE,
    ENABLE_DYNAMIC_SIZING,
    MIN_POSITION_SIZE, MAX_POSITION_SIZE,
    VIX_MODERATE_THRESHOLD,
    SIMULATION_MODE,               # ← NEW
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_SPREAD_BPS,
    SIM_MIN_FILL_DELAY_MS,
    SIM_MAX_FILL_DELAY_MS,
)

trade_tracker = TradeTracker()
notifier      = TelegramNotifier()


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _calc_dynamic_size(conf: float, vix: float) -> float:
    if not ENABLE_DYNAMIC_SIZING:
        return DEFAULT_POSITION_SIZE
    vix_factor = 0.7 if vix >= VIX_MODERATE_THRESHOLD else 1.0
    size = DEFAULT_POSITION_SIZE * conf * vix_factor
    return max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, size))


def _simulate_fill(market_price: float, side: str) -> float:
    """
    Very simple fill model:
        • add half‑spread
        • add slippage (bps = basis‑points)
    """
    spread = market_price * DEFAULT_SPREAD_BPS / 10_000
    mid    = market_price + (spread / 2 if side == "buy" else -spread / 2)
    slip   = market_price * DEFAULT_SLIPPAGE_BPS / 10_000
    filled = mid + (slip if side == "buy" else -slip)
    # random latency
    delay_ms = random.randint(SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS)
    time.sleep(delay_ms / 1000)
    return round(filled, 4), delay_ms


# ────────────────────────────────────────────────────────────
# Entry main
# ────────────────────────────────────────────────────────────
def handle_entry(market_data: dict):
    try:
        now = datetime.now()
        if now.hour == 15 and now.minute >= 30:
            bot_logger.info("[Entry] Blocked after 3:30 PM ET.")
            return

        signal = evaluate_trade_signal(market_data)
        if not signal.get("should_trade"):
            return

        confidence  = signal["confidence"]
        trade_type  = signal["trade_type"]          # 0 day / 1 swing
        trade_setup = signal["trade_setup"]

        # ── size & meta‑state
        vix   = get_current_vix() or 20
        size  = _calc_dynamic_size(confidence, vix)
        trade_setup["size"] = size

        meta_state = build_meta_state_for_entry(
            data_1m = signal["tf_1m"],
            data_5m = signal["tf_5m"],
            data_15m= signal["tf_15m"],
            data_1h = signal["tf_1h"],
            data_1d = signal["tf_1d"],
            confidence_score = confidence,
            trade_type       = trade_type,
            past_trades      = trade_tracker.get_open_trades(),
            long_term_data   = signal.get("long_term_data", {}),
            position_size    = size,
        )
        meta_agent.eval_mode()
        if meta_agent.select_action(meta_state) == 0:
            bot_logger.info("[Entry] Meta‑agent veto.")
            return

        # ── risk gate
        if not trade_tracker.can_place_trade():
            bot_logger.info("[Entry] Max open trades reached.")
            return

        # ──────────────────
        # LIVE vs SIM branch
        # ──────────────────
        if SIMULATION_MODE:
            m_price   = market_data["price"]
            side      = trade_setup.get("side", "buy")
            fill_px, latency_ms = _simulate_fill(m_price, side)
            order = {
                "id": f"sim_{int(time.time()*1000)}",
                "symbol": trade_setup.get("symbol","SPY"),
                "fill_price": fill_px,
                "status":"filled",
                "latency_ms": latency_ms,
            }
            slippage = abs(fill_px - m_price) / m_price
        else:
            t0 = time.time()
            order = execute_trade_with_retries(trade_setup)
            latency_ms = int((time.time() - t0)*1000)
            if not order:
                bot_logger.warning("[Entry] Real order failed.")
                return
            fill_px = order.get("fill_price", market_data["price"])
            slippage = abs(fill_px - market_data["price"]) / market_data["price"]

        # ── metrics + tracking
        log_trade_metrics(order["id"], latency_ms, slippage)
        order.update({
            "confidence": confidence,
            "size": size,
            "timestamp": now.isoformat(),
            "trade_type": trade_type,
            "meta_state": meta_state.tolist(),
            "meta_action": 1,          # ‘1’ = go / filled
            "option_symbol": trade_setup.get("option_symbol"),
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

        notifier.send_message(
            f"🟢 Entered {order['symbol']} size {size:.2%} "
            f"@ {fill_px} (slip {slippage*100:.2f} %)"
        )
        bot_logger.info(f"[Entry] Done: {order['symbol']} ({order['id']})")

    except Exception as exc:
        bot_logger.error(f"[Entry] {exc}")
        bot_logger.debug(traceback.format_exc())
        notifier.send_message(f"⚠️ Entry error\n{exc}")