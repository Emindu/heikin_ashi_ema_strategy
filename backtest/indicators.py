# backtest/indicators.py
"""Technical indicator calculations"""

import pandas as pd
import pandas_ta as ta


def prepare_indicators(df_entry: pd.DataFrame, df_trend: pd.DataFrame, 
                       ema_fast: int = 10, ema_slow: int = 30) -> tuple:
    """
    Calculate all required indicators on entry and trend dataframes.
    
    Returns:
        tuple: (df_entry, df_trend) with indicators added
    """
    # Calculate Heikin Ashi for Entry Data (for signal detection)
    if 'HA_open' not in df_entry.columns:
        ha_df = ta.ha(df_entry['open'], df_entry['high'], df_entry['low'], df_entry['close'])
        df_entry = df_entry.join(ha_df)
    
    # Calculate EMAs on trend (4H) data
    df_trend['fast_ema'] = ta.ema(df_trend['close'], length=ema_fast)
    df_trend['slow_ema'] = ta.ema(df_trend['close'], length=ema_slow)
    
    # Calculate ADX on trend data
    adx_data = ta.adx(df_trend['high'], df_trend['low'], df_trend['close'], length=14)
    if adx_data is not None and 'ADX_14' in adx_data.columns:
        df_trend['adx'] = adx_data['ADX_14']
    else:
        df_trend['adx'] = 0
    
    return df_entry, df_trend


def find_undecision_bar(row: pd.Series, threshold: float = 0.005) -> bool:
    """
    Check if a bar is an indecision bar (small body relative to range).
    
    Args:
        row: DataFrame row with OHLC data
        threshold: Maximum body/range ratio to qualify as indecision
        
    Returns:
        bool: True if indecision bar
    """
    body = abs(row['HA_close'] - row['HA_open'])
    range_ = row['HA_high'] - row['HA_low']
    
    if range_ == 0:
        return False
    
    return (body / range_) < threshold
