# exit.py

import pytz
import json
from datetime import datetime
from utils.trade_tracker import trade_tracker
from utils.trade_logger import log_trade_exit
from utils.logger import bot_logger
from trade_manager import close_trade
from meta.meta_agent import evaluate_exit_decision
from meta.reward_shaper import compute_shaped_reward
from meta.meta_state import build_meta_state_for_exit
from utils.telegram_notifier import TelegramNotifier
from config import META_LOG_PATH

eastern = pytz.timezone('US/Eastern')
notifier = TelegramNotifier()

def get_days_to_expiry(contract):
    try:
        expiry_str = contract.get("expiry")
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = datetime.now(eastern).date()
        return (expiry_date - today).days
    except Exception as e:
        bot_logger.warning(f"[Expiry Parse Error] {e}")
        return 0

def handle_exit():
    try:
        now = datetime.now(eastern)

        # ⏳ Force close day trades at 3:55 PM ET
        if now.hour == 15 and now.minute >= 55:
            closed_ids = []
            for trade in trade_tracker.get_open_trades():
                if trade.get("trade_type") == 0:
                    close_and_log_trade(trade, reason="Time-based exit (3:55 PM)")
                    closed_ids.append(str(trade.get("id", '?')))

            if closed_ids:
                notifier.send_message(
                    f"📉 [Auto Exit @ 3:55 PM ET]\nClosed {len(closed_ids)} day trade(s): {', '.join(closed_ids)}"
                )
            return

        # 📈 Dynamic exit evaluation
        for trade in trade_tracker.get_open_trades():
            exit_reason = should_exit_trade(trade)
            if exit_reason:
                close_and_log_trade(trade, reason=exit_reason)

    except Exception as e:
        bot_logger.error(f"[EXIT ERROR] Failed to handle exits: {str(e)}")

def should_exit_trade(trade):
    try:
        if evaluate_exit_decision(trade):
            return "Meta-agent signal"

        if trade.get("trade_type") == 1:
            dte = get_days_to_expiry(trade)
            if dte <= 1:
                return f"Contract near expiry (DTE={dte})"

        return None
    except Exception as e:
        bot_logger.error(f"[Exit Evaluation Error] {str(e)}")
        return None

def close_and_log_trade(trade, reason="Manual exit"):
    try:
        # 🧠 Compute reward and new state
        reward = compute_shaped_reward(trade)
        next_state = build_meta_state_for_exit(trade)

        trade.update({
            "shaped_reward": reward,
            "exit_reason": reason,
            "meta_next_state": next_state
        })

        close_trade(trade)
        trade_tracker.mark_trade_closed(trade.get("id"))
        log_trade_exit(trade)

        # 📢 Notify
        bot_logger.info(f"[EXIT] Closed trade {trade.get('id')} — Reason: {reason} — Reward: {reward:.3f}")
        notifier.send_message(f"🚪 Exited trade {trade.get('id')}\nReason: {reason}\nReward: {reward:.3f}")

        # 🧠 Store experience for meta-agent training
        state = trade.get("meta_state")
        action = trade.get("meta_action")
        if state and action is not None and next_state:
            with open(META_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": True
                }) + "\n")

    except Exception as e:
        bot_logger.error(f"[Trade Close Error] {e}")