# backtest/data.py
"""Data fetching functions with pagination support"""

import ccxt
import pandas as pd
from .config import TIMEFRAME_ENTRY, TIMEFRAME_TREND


def fetch_data_paginated(symbol: str, timeframe: str, total_candles: int) -> list:
    """Fetch more than 1000 candles by paginating backwards through time."""
    exchange = ccxt.binance()
    all_bars = []
    
    # Get timeframe in milliseconds for calculating offsets
    timeframe_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }.get(timeframe, 60 * 60 * 1000)
    
    print(f"Fetching {total_candles} candles for {symbol} ({timeframe})...")
    
    end_time = None
    
    while len(all_bars) < total_candles:
        try:
            if end_time is None:
                bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
            else:
                since = end_time - (1000 * timeframe_ms)
                bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
        if not bars:
            break
        
        if end_time is not None:
            bars = [b for b in bars if b[0] < end_time]
        
        if not bars:
            break
            
        all_bars = bars + all_bars
        end_time = bars[0][0]
        
        print(f"  Fetched {len(all_bars)} candles so far...")
        
    return all_bars[-total_candles:] if len(all_bars) > total_candles else all_bars


def fetch_data(symbol: str, limit: int) -> tuple:
    """Fetch entry (1H) and trend (4H) data for a symbol."""
    print(f"Fetching data for {symbol}...")
    
    # Fetch 1H data (paginated)
    bars_entry = fetch_data_paginated(symbol, TIMEFRAME_ENTRY, limit)
    df_entry = pd.DataFrame(bars_entry, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df_entry['time'] = pd.to_datetime(df_entry['time'], unit='ms')
    df_entry.set_index('time', inplace=True)

    # Fetch 4H data (paginated)
    bars_trend = fetch_data_paginated(symbol, TIMEFRAME_TREND, limit // 4 + 100)
    df_trend = pd.DataFrame(bars_trend, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df_trend['time'] = pd.to_datetime(df_trend['time'], unit='ms')
    df_trend.set_index('time', inplace=True)
    
    print(f"Loaded {len(df_entry)} 1H candles and {len(df_trend)} 4H candles.")
    print(f"  1H Data Range: {df_entry.index.min()} to {df_entry.index.max()}")
    print(f"  4H Data Range: {df_trend.index.min()} to {df_trend.index.max()}")
    
    return df_entry, df_trend
