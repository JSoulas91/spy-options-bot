# trading/entry.py
import time
from datetime import datetime
from utils.logger        import bot_logger
from utils.trade_tracker import TradeTracker
from utils.trade_logger  import log_trade
from utils.metrics_logger import log_trade_metrics
from meta.meta_agent     import meta_agent
from meta.meta_state     import build_meta_state_for_entry
from trade_manager       import execute_trade_with_retries
from strategy            import evaluate_trade_signal
from utils.telegram_notifier import TelegramNotifier
from utils.vix_utils     import get_current_vix
from config              import (
    DEFAULT_POSITION_SIZE,
    ENABLE_DYNAMIC_SIZING,
    MIN_POSITION_SIZE,
    MAX_POSITION_SIZE,
    VIX_MODERATE_THRESHOLD,
)

trade_tracker = TradeTracker()
notifier      = TelegramNotifier()

def _calc_dynamic_size(confidence: float, vix: float) -> float:
    """
    Simple dynamic‑sizing formula:
        size = base * confidence * vix_factor
    """
    if not ENABLE_DYNAMIC_SIZING:
        return DEFAULT_POSITION_SIZE
    # Penalize size if VIX elevated
    vix_factor = 0.7 if vix >= VIX_MODERATE_THRESHOLD else 1.0
    size = DEFAULT_POSITION_SIZE * confidence * vix_factor
    return max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, size))

def handle_entry(market_data):
    try:
        now = datetime.now()
        if now.hour == 15 and now.minute >= 30:
            bot_logger.info("[Entry] Blocked new entries after 3:30 PM ET.")
            return

        signal = evaluate_trade_signal(market_data)
        if not signal.get("should_trade"):
            return

        confidence  = signal["confidence"]
        trade_type  = signal["trade_type"]      # 0 = day, 1 = swing
        trade_setup = signal["trade_setup"]

        # Dynamic position size
        vix = get_current_vix() or 20
        position_size = _calc_dynamic_size(confidence, vix)
        trade_setup["size"] = position_size   # downstream expects 'size'

        # Meta‑state / agent
        meta_state = build_meta_state_for_entry(
            data_1m   = signal["tf_1m"],
            data_5m   = signal["tf_5m"],
            data_15m  = signal["tf_15m"],
            data_1h   = signal["tf_1h"],
            data_1d   = signal["tf_1d"],
            confidence_score = confidence,
            trade_type       = trade_type,
            past_trades      = trade_tracker.get_open_trades(),
            long_term_data   = signal.get("long_term_data", {}),
        )
        meta_agent.eval_mode()
        action = meta_agent.select_action(meta_state)
        if action == 0:
            bot_logger.info("[Entry] Meta‑agent rejected trade setup.")
            return

        # Max open trades check
        if not trade_tracker.can_place_trade():
            bot_logger.info("[Entry] Max open trades reached; skipping.")
            return

        # Execute trade
        t0 = time.time()
        order = execute_trade_with_retries(trade_setup)
        latency_ms = int((time.time() - t0) * 1000)

        if not order:
            bot_logger.warning("[Entry] Trade execution failed.")
            return

        # Calculate slippage (simple last‑price vs fill)
        slippage = abs((order.get("fill_price", 0) - market_data["price"])
                       / market_data["price"])

        # Metrics logging
        log_trade_metrics(order.get("id", "N/A"), latency_ms, slippage)

        # Enrich + log
        order.update({
            "confidence": confidence,
            "size": position_size,
            "timestamp": now.isoformat(),
            "trade_type": trade_type,
            "meta_state": meta_state.tolist(),
            "meta_action": int(action),
        })
        trade_tracker.log_trade(order, trade_type)
        log_trade({
            "timestamp": now.isoformat(),
            "action": "buy",
            "symbol": order.get("symbol", "UNKNOWN"),
            "size": position_size,
            "confidence": confidence,
            "trade_id": order.get("id", "N/A"),
        })

        notifier.send_message(
            f"Long {order.get('symbol')} | size {position_size:.2%} | conf {confidence:.2f}"
        )
        bot_logger.info(f"[Entry] Trade executed: {order.get('symbol')} size {position_size:.2%}")

    except Exception as e:
        bot_logger.exception(f"[Entry] Error: {e}")