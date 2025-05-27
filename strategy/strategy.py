import numpy as np
from strategy.confidence import compute_confidence_score
from strategy.market_regime import detect_market_regime
from strategy.volume_vwap import check_volume_vwap_confluence
from indicators.ema import calculate_ema
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.bollinger import calculate_bollinger_bands
from indicators.atr import calculate_atr
from utils.support_resistance import detect_key_levels

class Strategy:
    def __init__(self, aggressive_mode=False):
        self.aggressive_mode = aggressive_mode

    def evaluate_trade(self, df, current_price, trade_type, sentiment_score, news_event_risk, premarket_trend):
        """
        Main unified strategy logic. Filters out trades based on:
        - Technicals
        - Sentiment
        - Volume/VWAP
        - Market Regime
        - Support/Resistance
        - Confidence scoring
        - Economic blackout zones
        """

        # === Prevent trades near economic events ===
        if news_event_risk:
            return False, "Blackout zone: economic event"

        # === Technical Indicators ===
        ema_50 = calculate_ema(df, 50)
        ema_200 = calculate_ema(df, 200)
        rsi = calculate_rsi(df)
        macd, signal = calculate_macd(df)
        upper_band, lower_band = calculate_bollinger_bands(df)
        atr = calculate_atr(df)

        # === Basic Price Action Rules ===
        price_above_ema = current_price > ema_50.iloc[-1] and current_price > ema_200.iloc[-1]
        macd_bullish = macd.iloc[-1] > signal.iloc[-1]
        rsi_ok = rsi.iloc[-1] < 70 and rsi.iloc[-1] > 30

        # === Volume/VWAP confluence ===
        if not check_volume_vwap_confluence(df):
            return False, "No volume/VWAP support"

        # === Market Regime Check ===
        market_ok = detect_market_regime(df)

        # === Key Support/Resistance ===
        key_levels = detect_key_levels(df)
        near_resistance = current_price >= key_levels['resistance'] * 0.99 if trade_type == 'call' else False
        near_support = current_price <= key_levels['support'] * 1.01 if trade_type == 'put' else False

        if (trade_type == 'call' and near_resistance) or (trade_type == 'put' and near_support):
            return False, "Too close to S/R level"

        # === Sentiment Weight ===
        if sentiment_score < -0.3:
            return False, "Negative sentiment filter"

        # === Confidence Score ===
        score = compute_confidence_score(
            current_price=current_price,
            ema_50=ema_50.iloc[-1],
            ema_200=ema_200.iloc[-1],
            macd=macd.iloc[-1],
            signal=signal.iloc[-1],
            rsi=rsi.iloc[-1],
            atr=atr.iloc[-1],
            sentiment_score=sentiment_score,
            market_regime=market_ok,
            volume_vwap_pass=True,
            support_resistance_pass=True,
            premarket_trend=premarket_trend,
            trade_type=trade_type,
            aggressive_mode=self.aggressive_mode
        )

        # === Decision Threshold ===
        if score >= 0.75:
            return True, f"High confidence ({score:.2f})"
        elif self.aggressive_mode and score >= 0.6:
            return True, f"Aggressive mode entry ({score:.2f})"
        else:
            return False, f"Low confidence ({score:.2f})"
