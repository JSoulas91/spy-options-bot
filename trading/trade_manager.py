import alpaca_trade_api as tradeapi
import logging
import time
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

logger = logging.getLogger(__name__)

class TradeManager:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

    def get_account(self):
        try:
            return self.api.get_account()
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None

    def get_positions(self):
        try:
            return self.api.list_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def close_position(self, symbol):
        try:
            self.api.close_position(symbol)
            logger.info(f"Closed position for {symbol}")
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")

    def submit_order(self, symbol, qty, side, type='market', time_in_force='gtc'):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force
            )
            logger.info(f"{side.capitalize()} order submitted for {qty} shares of {symbol}")
            return order
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return None

    def check_existing_order(self, symbol):
        try:
            orders = self.api.list_orders(status='open', symbols=[symbol])
            return any(order.symbol == symbol for order in orders)
        except Exception as e:
            logger.error(f"Failed to check existing orders: {e}")
            return False

    def cancel_all_orders(self):
        try:
            self.api.cancel_all_orders()
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
