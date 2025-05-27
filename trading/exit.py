from utils.logger import log_info

def evaluate_exit_conditions(position, market_data, indicators, confidence_score, trailing_stop_enabled=True):
    exit_signal = False
    reason = ""

    # Example: Exit if confidence drops too low
    if confidence_score < 0.3:
        exit_signal = True
        reason = "Low confidence score"

    # Example: Exit if stop loss or trailing stop hit
    if trailing_stop_enabled and position.get("trailing_stop_hit", False):
        exit_signal = True
        reason = "Trailing stop hit"

    # Example: Exit if technical reversal detected
    if indicators.get("exit_signal", False):
        exit_signal = True
        reason = "Technical reversal signal"

    if exit_signal:
        log_info(f"Exit Signal triggered. Reason: {reason}")
        return True
    return False
