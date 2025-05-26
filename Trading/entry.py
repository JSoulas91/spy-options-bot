from strategy import generate_trade_signal
from utils.logger import log_info

def evaluate_entry_signals(market_data, indicators, sentiment, confidence_score):
    signal = generate_trade_signal(market_data, indicators, sentiment, confidence_score)
    
    if signal == "buy_call":
        log_info("Entry Signal: BUY CALL triggered.")
        return "CALL"
    elif signal == "buy_put":
        log_info("Entry Signal: BUY PUT triggered.")
        return "PUT"
    else:
        log_info("No valid entry signal.")
        return None
