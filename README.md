# SPY Options Bot

An advanced, fully automated SPY options trading bot that combines technical analysis, machine learning, sentiment analysis, economic awareness, and smart risk management — deployable entirely from your phone.

---

## 🚀 Features

- 🔁 Real-time automated trading (day & swing trades)
- 📊 Technical indicators: EMA, RSI, MACD, Bollinger Bands, VWAP, ATR, volume spikes
- 🤖 Machine Learning: predictive modeling + daily retraining
- 🧠 NLP Sentiment Analysis on news & social media
- 📅 Economic calendar awareness (auto-blackout around events)
- ⚠️ Adaptive risk: ATR-based stop-loss, tiered take-profit, dynamic position sizing
- 🧩 Confidence scoring system to filter trades
- 🧭 Market regime detection to adjust strategy
- 📈 Premarket scanning with news, key levels, and volume tracking
- 📡 Telegram integration for alerts and performance dashboard
- 💡 Strategy auto-off after key trades or news-based risk triggers

---

## 📁 Folder Structure

```bash
spy-options-bot/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── utils/
│   ├── helpers.py
│   ├── indicators.py
│   └── logger.py
├── strategy/
│   └── strategy.py
├── trading/
│   ├── entry.py
│   ├── exit.py
│   └── trade_manager.py
├── ml/
│   ├── model.py
│   └── retrain.py
├── sentiment/
│   └── sentiment_analyzer.py
├── filters/
│   ├── confidence_filter.py
│   ├── event_filter.py
│   └── volume_vwap_filter.py
└── dashboard/
    └── telegram_reporter.py
