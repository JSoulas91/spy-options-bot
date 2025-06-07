# strategy/strategy.py  – uses direction + confidence from PPO
# (imports unchanged at top – omitted for brevity)
#  …  existing helper functions keep

MIN_META_CONF = MIN_META_CONFIDENCE      # from .env

# ——————————————————————————————————————————
def evaluate_trade(position: Dict, market_data: Dict) -> str:
    try:
        # …  (indicator merge code exactly as before) …

        # —— Meta‑agent call ————————————————————
        meta_state  = normalize_meta_state({...})
        dir_idx, conf_est, *_ = meta_agent.policy.act(meta_state)   # <- NEW

        # Reject if PPO confidence low
        if conf_est < MIN_META_CONF:
            return "exit"
        # Interpret direction
        if dir_idx == 0:           # PPO suggests immediate exit
            return "exit"
        tight_exit = (dir_idx == 2)

        # —— confidence / VIX filters (unchanged) ——
        # … same logic …

        # Tight‑exit option from PPO
        if tight_exit and confidence < (threshold + 0.05):
            return "exit"

        # —— rest of day/swing exit logic stays the same ——
        # …

        return "hold"
    except Exception as e:
        logger.error(f"[Strategy] {e}\n{traceback.format_exc()}")
        return "hold"