from trading.entry import evaluate_entry_signals
from trading.exit import evaluate_exit_conditions
from utils.broker import place_order, close_position
from utils.logger import log_info

class TradeManager:
    def __init__(self):
        self.active_position = None

    def check_for_entry(self, market_data, indicators, sentiment, confidence_score):
        if not self.active_position:
            entry_side = evaluate_entry_signals(market_data, indicators, sentiment, confidence_score)
            if entry_side:
                self.active_position = place_order(entry_side)
                log_info(f"Opened new position: {entry_side}")

    def check_for_exit(self, market_data, indicators, confidence_score):
        if self.active_position:
            if evaluate_exit_conditions(self.active_position, market_data, indicators, confidence_score):
                close_position(self.active_position)
                log_info(f"Closed position: {self.active_position['type']}")
                self.active_position = None
