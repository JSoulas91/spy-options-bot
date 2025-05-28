import pytz
from datetime import datetime
from alpaca_trade_api.rest import REST, TimeFrame
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from helpers import is_day_trade, is_swing_trade

# Set timezone to Eastern
eastern = pytz.timezone("US/Eastern")

# Alpaca client
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

def get_current_time_et():
    return datetime.now(eastern)

def is_market_open():
    clock = alpaca.get_clock()
    return clock.is_open

def should_enter_day_trade():
    now = get_current_time_et()
    return now.hour < 15 or (now.hour == 15 and now.minute < 30)

def should_exit_day_trades():
    now = get_current_time_et()
    return now.hour == 15 and now.minute >= 55  # 3:55 PM ET

def place_order(symbol, qty, side, type="market", time_in_force="gtc"):
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
        )
        print(f"Order placed: {side.upper()} {qty} {symbol}")
        return {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'entry_time': datetime.utcnow().isoformat(),  # Required for trade classification
            'order_id': order.id
        }
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def close_position(symbol):
    try:
        alpaca.close_position(symbol)
        print(f"Closed position: {symbol}")
    except Exception as e:
        print(f"Error closing position: {e}")

def manage_open_positions(positions):
    for pos in positions:
        symbol = pos['symbol']
        if is_day_trade(pos) and should_exit_day_trades():
            print(f"Day trade cutoff hit, closing position: {symbol}")
            close_position(symbol)

def get_open_positions():
    try:
        positions = alpaca.list_positions()
        return [p._raw for p in positions if p.qty != '0']
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []