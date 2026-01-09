# backtest/config.py
"""Configuration settings for backtesting"""

# --- Symbol & Timeframe ---
SYMBOL = 'BTC/USDT'
TIMEFRAME_ENTRY = '1h'
TIMEFRAME_TREND = '4h'
HISTORY_LIMIT = 10000

# --- Strategy Parameters ---
INDECISION_THRESHOLD = 0.005
ADX_THRESHOLD = 25  # Minimum ADX for trend strength

# --- Futures Trading Settings ---
LEVERAGE = 10
TRADING_FEE = 0.0004  # 0.04% per trade (taker fee)
RISK_PER_TRADE = 0.05  # Risk 5% of portfolio per trade
STOP_LOSS_ATR_MULT = 1.5  # Stop-loss distance: 1.5x ATR

# --- Optimization Parameter Ranges ---
PARAM_EMA_FAST = [5, 10, 15, 20]
PARAM_EMA_SLOW = [20, 30, 40, 50]
PARAM_INDECISION = [0.003, 0.005, 0.007, 0.01]
PARAM_ADX = [20, 25, 30]
