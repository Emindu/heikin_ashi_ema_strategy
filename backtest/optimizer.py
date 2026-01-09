# backtest/optimizer.py
"""Parameter optimization via grid search"""

from itertools import product
import pandas_ta as ta

from .config import (
    SYMBOL, HISTORY_LIMIT,
    PARAM_EMA_FAST, PARAM_EMA_SLOW, PARAM_INDECISION, PARAM_ADX
)
from .data import fetch_data
from .strategy import run_backtest


def run_optimization():
    """
    Run grid search optimization over parameter ranges.
    """
    print("=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Fetch data once
    df_entry_raw, df_trend_raw = fetch_data(SYMBOL, HISTORY_LIMIT)
    
    # Calculate Heikin Ashi once (doesn't change with parameters)
    ha_df = ta.ha(df_entry_raw['open'], df_entry_raw['high'], 
                  df_entry_raw['low'], df_entry_raw['close'])
    df_entry_raw = df_entry_raw.join(ha_df)
    
    results = []
    total_combinations = len(PARAM_EMA_FAST) * len(PARAM_EMA_SLOW) * len(PARAM_INDECISION) * len(PARAM_ADX)
    current = 0
    
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    for ema_fast, ema_slow, indecision, adx_thresh in product(
            PARAM_EMA_FAST, PARAM_EMA_SLOW, PARAM_INDECISION, PARAM_ADX):
        
        # Skip invalid combinations (fast >= slow)
        if ema_fast >= ema_slow:
            continue
            
        current += 1
        
        metrics = run_backtest(df_entry_raw, df_trend_raw, ema_fast, ema_slow, indecision, adx_thresh)
        
        results.append({
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'indecision': indecision,
            'adx_threshold': adx_thresh,
            **metrics
        })
    
    # Sort by final portfolio (best first)
    results.sort(key=lambda x: x['final_portfolio'], reverse=True)
    
    # Print top 10 results
    _print_results(results)


def _print_results(results: list):
    """Print optimization results table."""
    print("\n" + "=" * 95)
    print("TOP 10 PARAMETER COMBINATIONS (by Final Portfolio)")
    print("=" * 95)
    print(f"{'Rank':<5} | {'Fast':<5} | {'Slow':<5} | {'Indec.':<8} | {'ADX':<4} | {'Trades':<7} | {'WinRate':<8} | {'Return%':<10} | {'Portfolio':<10}")
    print("-" * 95)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<5} | {r['ema_fast']:<5} | {r['ema_slow']:<5} | {r['indecision']:<8.4f} | {r['adx_threshold']:<4} | {r['total_trades']:<7} | {r['win_rate']:<8.2f}% | {r['return_pct']:<10.2f}% | ${r['final_portfolio']:<10.2f}")
    
    print("-" * 95)
    
    if results:
        best = results[0]
        print(f"\n🏆 BEST PARAMETERS: EMA_FAST={best['ema_fast']}, EMA_SLOW={best['ema_slow']}, INDECISION={best['indecision']}, ADX={best['adx_threshold']}")
        print(f"   → {best['total_trades']} trades, {best['win_rate']:.2f}% win rate, ${best['final_portfolio']:.2f} final portfolio")
