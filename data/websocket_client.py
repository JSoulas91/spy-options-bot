# data/websocket_client.py
from alpaca_trade_api.stream import Stream
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

def start_websocket(on_bar_callback):
    stream = Stream(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

    @stream.on_bar("SPY")
    async def handle_bar(bar):
        await on_bar_callback(bar)

    stream.run()