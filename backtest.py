import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
import argparse

# --- Configuration ---
SYMBOL = 'ATOM/USDT'
TIMEFRAME_ENTRY = '1h'
TIMEFRAME_TREND = '4h'
HISTORY_LIMIT = 10000  # Increased for optimization
INDECISION_THRESHOLD = 0.005
ADX_THRESHOLD = 25  # Minimum ADX for trend strength

# --- Futures Trading Settings ---
LEVERAGE = 10  # 10x leverage
TRADING_FEE = 0.0004  # Binance Futures taker fee: 0.04% per trade (entry + exit = 0.08% total)
RISK_PER_TRADE = 0.05  # Risk 5% of portfolio per trade
STOP_LOSS_ATR_MULT = 1.5  # Stop-loss distance: 1.5x ATR from entry

# --- Optimization Parameter Ranges ---
PARAM_EMA_FAST = [5, 10, 15, 20]
PARAM_EMA_SLOW = [20, 30, 40, 50]
PARAM_INDECISION = [0.003, 0.005, 0.007, 0.01]
PARAM_ADX = [20, 25, 30]  # ADX threshold values to test

def fetch_data_paginated(symbol, timeframe, total_candles):
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
    }.get(timeframe, 60 * 60 * 1000)  # Default to 1h
    
    print(f"Fetching {total_candles} candles for {symbol} ({timeframe})...")
    
    # Start with most recent data (since=None means "from now")
    # We'll work backwards from here
    end_time = None  # Will be set after first fetch
    
    while len(all_bars) < total_candles:
        try:
            if end_time is None:
                # First fetch - get most recent data
                bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
            else:
                # Subsequent fetches - get data BEFORE our current oldest bar
                # Calculate 'since' to be 1000 bars before end_time
                since = end_time - (1000 * timeframe_ms)
                bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
        if not bars:
            break
        
        # Filter to only keep bars BEFORE our current end_time (avoid duplicates)
        if end_time is not None:
            bars = [b for b in bars if b[0] < end_time]
        
        if not bars:
            break
            
        # Prepend older data to our collection
        all_bars = bars + all_bars
        
        # Update end_time to the oldest bar we have
        end_time = bars[0][0]
        
        print(f"  Fetched {len(all_bars)} candles so far...")
        
    # Trim to exact count (keep most recent)
    return all_bars[-total_candles:] if len(all_bars) > total_candles else all_bars

def fetch_data(symbol, limit):
    print(f"Fetching data for {symbol}...")
    
    # Fetch 1H data (paginated)
    bars_entry = fetch_data_paginated(symbol, TIMEFRAME_ENTRY, limit)
    df_entry = pd.DataFrame(bars_entry, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df_entry['time'] = pd.to_datetime(df_entry['time'], unit='ms')
    df_entry.set_index('time', inplace=True)

    # Fetch 4H data (paginated)
    # For 4H, we need fewer candles since each covers 4 hours
    bars_trend = fetch_data_paginated(symbol, TIMEFRAME_TREND, limit // 4 + 100)
    df_trend = pd.DataFrame(bars_trend, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df_trend['time'] = pd.to_datetime(df_trend['time'], unit='ms')
    df_trend.set_index('time', inplace=True)
    
    print(f"Loaded {len(df_entry)} 1H candles and {len(df_trend)} 4H candles.")
    print(f"  1H Data Range: {df_entry.index.min()} to {df_entry.index.max()}")
    print(f"  4H Data Range: {df_trend.index.min()} to {df_trend.index.max()}")
    return df_entry, df_trend

def prepare_indicators(df_entry, df_trend, ema_fast=10, ema_slow=30):
    # 1. Calculate Heikin Ashi for Entry Data (for signal detection) - only if not already present
    if 'HA_open' not in df_entry.columns:
        ha_df = ta.ha(df_entry['open'], df_entry['high'], df_entry['low'], df_entry['close'])
        df_entry = df_entry.join(ha_df)

    # 2. Calculate EMAs on 4H Data (parameterized)
    df_trend['slow_ema'] = df_trend['close'].rolling(window=ema_slow).mean()
    df_trend['fast_ema'] = df_trend['close'].rolling(window=ema_fast).mean()
    
    # 3. Calculate ADX on 4H Data for trend strength
    adx_df = ta.adx(df_trend['high'], df_trend['low'], df_trend['close'], length=14)
    if adx_df is not None and 'ADX_14' in adx_df.columns:
        df_trend['adx'] = adx_df['ADX_14']
    else:
        df_trend['adx'] = 0  # Fallback if ADX calculation fails
    
    # 4. Determine Trend
    df_trend['trend'] = 0
    df_trend.loc[df_trend['fast_ema'] > df_trend['slow_ema'], 'trend'] = 1
    df_trend.loc[df_trend['fast_ema'] < df_trend['slow_ema'], 'trend'] = -1
    
    return df_entry, df_trend

def find_undecision_bar(row, indecision_threshold):
    if (row['HA_high'] > row['HA_close']) and (row['HA_low'] < row['HA_open']):
        high_open_diff = (row['HA_high'] - row['HA_open']) / row['HA_high'] if row['HA_high'] != 0 else 0
        low_close_diff = (row['HA_close'] - row['HA_low']) / row['HA_close'] if row['HA_close'] != 0 else 0
        
        if high_open_diff > indecision_threshold and low_close_diff > indecision_threshold:
            return True
    return False

def run_backtest_and_plot():
    df_entry, df_trend = fetch_data(SYMBOL, HISTORY_LIMIT)
    
    # Remove duplicate indices and sort
    df_entry = df_entry[~df_entry.index.duplicated(keep='first')].sort_index()
    df_trend = df_trend[~df_trend.index.duplicated(keep='first')].sort_index()
    
    df_entry, df_trend = prepare_indicators(df_entry, df_trend)

    trades = []
    position = None 
    entry_price = 0
    
    # For Visualization: Create columns to store signals on the timeframe
    df_entry['buy_signal'] = float('nan')
    df_entry['sell_signal'] = float('nan')
    df_entry['buy_signal_plot'] = float('nan')
    df_entry['sell_signal_plot'] = float('nan')
    df_entry['exit_signal'] = float('nan')
    
    # Align 4H EMAs and ADX to 1H Timeframe for plotting
    # We use reindex and ffill (forward fill) to show the 4H values on every 1H bar
    df_aligned = df_trend.reindex(df_entry.index, method='ffill')
    df_entry['trend_fast_ema'] = df_aligned['fast_ema']
    df_entry['trend_slow_ema'] = df_aligned['slow_ema']
    df_entry['trend_adx'] = df_aligned['adx']

    # Calculate ATR for visual offset
    df_entry['atr'] = ta.atr(df_entry['high'], df_entry['low'], df_entry['close'], length=14)
    # Fill NaN ATR values (start of data) with backfill or 0 to avoid errors
    df_entry['atr'] = df_entry['atr'].bfill()
    
    # Define offset multiplier
    ATR_MULTIPLIER = 1.0

    print("Running simulation...")
    
    for i in range(30, len(df_entry) - 1):
        current_bar = df_entry.iloc[i]
        prev_bar = df_entry.iloc[i-1]
        current_time = df_entry.index[i]
        
        # Get trend and ADX from the aligned columns
        ema_fast = current_bar['trend_fast_ema']
        ema_slow = current_bar['trend_slow_ema']
        adx_value = current_bar['trend_adx']
        
        # Determine trend based on aligned EMAs
        if pd.isna(ema_fast) or pd.isna(ema_slow): continue
        current_trend = 1 if ema_fast > ema_slow else -1
        
        # Check ADX filter (skip if trend is weak)
        adx_ok = (not pd.isna(adx_value)) and (adx_value >= ADX_THRESHOLD)

        # LOGIC
        if position is None:
            is_indecision = find_undecision_bar(prev_bar, INDECISION_THRESHOLD)
            
            # LONG ENTRY (only if ADX confirms strong trend)
            if current_trend == 1 and is_indecision and adx_ok:
                if (current_bar['HA_close'] > current_bar['HA_open']) and \
                   (abs(current_bar['HA_open'] - current_bar['HA_low']) < 0.0001 * current_bar['HA_open']):
                    
                    position = 'LONG'
                    entry_price = df_entry.iloc[i+1]['open']
                    atr_value = current_bar['atr']
                    
                    # Calculate stop-loss (below entry by ATR multiplier)
                    stop_loss = entry_price - (atr_value * STOP_LOSS_ATR_MULT)
                    stop_distance_pct = (entry_price - stop_loss) / entry_price
                    
                    # Calculate position size based on risk
                    # position_size = risk / stop_distance (as fraction of portfolio)
                    if stop_distance_pct > 0:
                        position_size = RISK_PER_TRADE / (stop_distance_pct * LEVERAGE)
                        position_size = min(position_size, 1.0)  # Cap at 100% of portfolio
                    else:
                        position_size = RISK_PER_TRADE
                    
                    # Store Plotting Info
                    plot_y = df_entry.iloc[i]['low'] - (atr_value * ATR_MULTIPLIER)
                    df_entry.loc[df_entry.index[i+1], 'buy_signal'] = entry_price
                    df_entry.loc[df_entry.index[i+1], 'buy_signal_plot'] = plot_y
                    

            # SHORT ENTRY (only if ADX confirms strong trend)
            elif current_trend == -1 and is_indecision and adx_ok:
                if (current_bar['HA_close'] < current_bar['HA_open']) and \
                   (abs(current_bar['HA_high'] - current_bar['HA_open']) < 0.0001 * current_bar['HA_open']):
                    
                    position = 'SHORT'
                    entry_price = df_entry.iloc[i+1]['open']
                    atr_value = current_bar['atr']
                    
                    # Calculate stop-loss (above entry by ATR multiplier)
                    stop_loss = entry_price + (atr_value * STOP_LOSS_ATR_MULT)
                    stop_distance_pct = (stop_loss - entry_price) / entry_price
                    
                    # Calculate position size based on risk
                    if stop_distance_pct > 0:
                        position_size = RISK_PER_TRADE / (stop_distance_pct * LEVERAGE)
                        position_size = min(position_size, 1.0)
                    else:
                        position_size = RISK_PER_TRADE
                    
                    # Store Plotting Info
                    plot_y = df_entry.iloc[i]['high'] + (atr_value * ATR_MULTIPLIER)
                    df_entry.loc[df_entry.index[i+1], 'sell_signal'] = entry_price
                    df_entry.loc[df_entry.index[i+1], 'sell_signal_plot'] = plot_y

        # EXITS
        elif position == 'LONG':
            # Check stop-loss first (using low price as worst case)
            current_low = df_entry.iloc[i]['low']
            stopped_out = current_low <= stop_loss
            signal_exit = current_bar['HA_close'] < current_bar['HA_open']
            
            if stopped_out or signal_exit:
                if stopped_out:
                    exit_price = stop_loss  # Assume fill at stop price
                else:
                    exit_price = df_entry.iloc[i+1]['open']
                
                # Calculate PnL with leverage (already factored into position_size)
                raw_pnl = (exit_price - entry_price) / entry_price * 100
                leveraged_pnl = raw_pnl * LEVERAGE
                total_fees = TRADING_FEE * 2 * 100
                trade_pnl = leveraged_pnl - total_fees
                
                # Portfolio PnL is trade PnL * position_size
                portfolio_pnl = trade_pnl * position_size
                
                trades.append({
                    'type': 'LONG' + (' (SL)' if stopped_out else ''),
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': portfolio_pnl,
                    'time': df_entry.index[i+1] if not stopped_out else df_entry.index[i]
                })
                df_entry.loc[df_entry.index[i+1], 'exit_signal'] = exit_price
                position = None

        elif position == 'SHORT':
            # Check stop-loss first (using high price as worst case)
            current_high = df_entry.iloc[i]['high']
            stopped_out = current_high >= stop_loss
            signal_exit = current_bar['HA_close'] > current_bar['HA_open']
            
            if stopped_out or signal_exit:
                if stopped_out:
                    exit_price = stop_loss  # Assume fill at stop price
                else:
                    exit_price = df_entry.iloc[i+1]['open']
                
                # Calculate PnL with leverage
                raw_pnl = (entry_price - exit_price) / entry_price * 100
                leveraged_pnl = raw_pnl * LEVERAGE
                total_fees = TRADING_FEE * 2 * 100
                trade_pnl = leveraged_pnl - total_fees
                
                # Portfolio PnL is trade PnL * position_size
                portfolio_pnl = trade_pnl * position_size
                
                trades.append({
                    'type': 'SHORT' + (' (SL)' if stopped_out else ''),
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': portfolio_pnl,
                    'time': df_entry.index[i+1] if not stopped_out else df_entry.index[i]
                })
                df_entry.loc[df_entry.index[i+1], 'exit_signal'] = exit_price
                position = None

    # --- METRICS CALCULATION ---
    initial_balance = 100
    current_balance = initial_balance
    winning_trades = 0

    print("\n" + "="*70)
    print(f"{'Type':<6} | {'Date':<20} | {'Entry':<10} | {'Exit':<10} | {'PnL (%)':<8}")
    print("-" * 70)

    for trade in trades:
        pnl = trade['pnl']
        if pnl > 0:
            winning_trades += 1
        
        # Print Trade Detail
        print(f"{trade['type']:<10} | {str(trade['time']):<20} | {trade['entry']:<10.4f} | {trade['exit']:<10.4f} | {pnl:<8.2f}%")

        # Compounding balance (pnl already includes position size)
        current_balance = current_balance * (1 + pnl / 100)
    
    print("-" * 70)

    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print("-" * 30)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Initial Portfolio: ${initial_balance:.2f}")
    print(f"Final Portfolio:   ${current_balance:.2f}")
    print("-" * 30)
    
    # --- PLOTTING ---
    print("Generating Plot...")
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # 1. Main Candlestick (Heikin Ashi)
    fig.add_trace(go.Candlestick(
        x=df_entry.index,
        open=df_entry['HA_open'], high=df_entry['HA_high'],
        low=df_entry['HA_low'], close=df_entry['HA_close'],
        name='Heikin Ashi (1H)'
    ))

    # 2. Trend EMAs (From 4H, overlaid on 1H)
    fig.add_trace(go.Scatter(
        x=df_entry.index, y=df_entry['trend_fast_ema'], 
        line=dict(color='orange', width=1), name='Fast EMA (4H Trend)'
    ))
    fig.add_trace(go.Scatter(
        x=df_entry.index, y=df_entry['trend_slow_ema'], 
        line=dict(color='blue', width=1), name='Slow EMA (4H Trend)'
    ))

    # 3. Buy Signals (Green Triangles) - Updated to use Offset
    fig.add_trace(go.Scatter(
        x=df_entry.index, y=df_entry['buy_signal_plot'],
        mode='markers', 
        marker=dict(symbol='triangle-up', size=14, color='green'),
        name='Buy Signal',
        hovertext=df_entry['buy_signal'].apply(lambda x: f"Buy Price: {x:.2f}" if pd.notnull(x) else ""),
        hoverinfo='text+x'
    ))

    # 4. Sell Signals (Red Triangles) - Updated to use Offset
    fig.add_trace(go.Scatter(
        x=df_entry.index, y=df_entry['sell_signal_plot'],
        mode='markers', 
        marker=dict(symbol='triangle-down', size=14, color='red'),
        name='Sell Signal',
        hovertext=df_entry['sell_signal'].apply(lambda x: f"Sell Price: {x:.2f}" if pd.notnull(x) else ""),
        hoverinfo='text+x'
    ))

    # 5. Exits (Black X)
    fig.add_trace(go.Scatter(
        x=df_entry.index, y=df_entry['exit_signal'],
        mode='markers', marker=dict(symbol='x', size=10, color='white', line=dict(color='white', width=2)),
        name='Trade Exit'
    ))

    fig.update_layout(
        title=f'Backtest Strategy Visualization: {SYMBOL}',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    fig.show()


def run_backtest(df_entry, df_trend, ema_fast, ema_slow, indecision_threshold, adx_threshold=25):
    """
    Run backtest with specific parameters and return metrics (no plotting).
    """
    # Make deep copies to avoid modifying original data
    df_entry = df_entry.copy(deep=True)
    df_trend = df_trend.copy(deep=True)
    
    # Remove any duplicate indices and sort to ensure monotonic order
    df_entry = df_entry[~df_entry.index.duplicated(keep='first')].sort_index()
    df_trend = df_trend[~df_trend.index.duplicated(keep='first')].sort_index()
    
    # Prepare indicators with parameters
    df_entry, df_trend = prepare_indicators(df_entry, df_trend, ema_fast, ema_slow)
    
    # Calculate ATR for stop-loss
    df_entry['atr'] = ta.atr(df_entry['high'], df_entry['low'], df_entry['close'], length=14)
    df_entry['atr'] = df_entry['atr'].bfill()
    
    trades = []
    position = None 
    entry_price = 0
    stop_loss = 0
    position_size = 0
    
    # Align 4H EMAs and ADX to 1H Timeframe
    df_aligned = df_trend.reindex(df_entry.index, method='ffill')
    df_entry['trend_fast_ema'] = df_aligned['fast_ema']
    df_entry['trend_slow_ema'] = df_aligned['slow_ema']
    df_entry['trend_adx'] = df_aligned['adx']
    
    start_idx = max(30, ema_slow + 5)
    
    for i in range(start_idx, len(df_entry) - 1):
        current_bar = df_entry.iloc[i]
        prev_bar = df_entry.iloc[i-1]
        
        ema_fast_val = current_bar['trend_fast_ema']
        ema_slow_val = current_bar['trend_slow_ema']
        adx_value = current_bar['trend_adx']
        
        if pd.isna(ema_fast_val) or pd.isna(ema_slow_val): 
            continue
        current_trend = 1 if ema_fast_val > ema_slow_val else -1
        
        adx_ok = (not pd.isna(adx_value)) and (adx_value >= adx_threshold)

        if position is None:
            is_indecision = find_undecision_bar(prev_bar, indecision_threshold)
            atr_value = current_bar['atr'] if not pd.isna(current_bar['atr']) else 0
            
            # LONG ENTRY
            if current_trend == 1 and is_indecision and adx_ok:
                if (current_bar['HA_close'] > current_bar['HA_open']) and \
                   (abs(current_bar['HA_open'] - current_bar['HA_low']) < 0.0001 * current_bar['HA_open']):
                    position = 'LONG'
                    entry_price = df_entry.iloc[i+1]['open']
                    stop_loss = entry_price - (atr_value * STOP_LOSS_ATR_MULT)
                    stop_distance_pct = (entry_price - stop_loss) / entry_price if entry_price > 0 else 0
                    if stop_distance_pct > 0:
                        position_size = min(RISK_PER_TRADE / (stop_distance_pct * LEVERAGE), 1.0)
                    else:
                        position_size = RISK_PER_TRADE

            # SHORT ENTRY
            elif current_trend == -1 and is_indecision and adx_ok:
                if (current_bar['HA_close'] < current_bar['HA_open']) and \
                   (abs(current_bar['HA_high'] - current_bar['HA_open']) < 0.0001 * current_bar['HA_open']):
                    position = 'SHORT'
                    entry_price = df_entry.iloc[i+1]['open']
                    stop_loss = entry_price + (atr_value * STOP_LOSS_ATR_MULT)
                    stop_distance_pct = (stop_loss - entry_price) / entry_price if entry_price > 0 else 0
                    if stop_distance_pct > 0:
                        position_size = min(RISK_PER_TRADE / (stop_distance_pct * LEVERAGE), 1.0)
                    else:
                        position_size = RISK_PER_TRADE

        # EXITS
        elif position == 'LONG':
            current_low = df_entry.iloc[i]['low']
            stopped_out = current_low <= stop_loss
            signal_exit = current_bar['HA_close'] < current_bar['HA_open']
            
            if stopped_out or signal_exit:
                exit_price = stop_loss if stopped_out else df_entry.iloc[i+1]['open']
                raw_pnl = (exit_price - entry_price) / entry_price * 100
                leveraged_pnl = raw_pnl * LEVERAGE
                total_fees = TRADING_FEE * 2 * 100
                trade_pnl = leveraged_pnl - total_fees
                portfolio_pnl = trade_pnl * position_size
                trades.append(portfolio_pnl)
                position = None

        elif position == 'SHORT':
            current_high = df_entry.iloc[i]['high']
            stopped_out = current_high >= stop_loss
            signal_exit = current_bar['HA_close'] > current_bar['HA_open']
            
            if stopped_out or signal_exit:
                exit_price = stop_loss if stopped_out else df_entry.iloc[i+1]['open']
                raw_pnl = (entry_price - exit_price) / entry_price * 100
                leveraged_pnl = raw_pnl * LEVERAGE
                total_fees = TRADING_FEE * 2 * 100
                trade_pnl = leveraged_pnl - total_fees
                portfolio_pnl = trade_pnl * position_size
                trades.append(portfolio_pnl)
                position = None

    # Calculate metrics
    initial_balance = 100
    current_balance = initial_balance
    winning_trades = 0
    
    for pnl in trades:
        if pnl > 0:
            winning_trades += 1
        current_balance = current_balance * (1 + pnl / 100)
    
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'final_portfolio': current_balance,
        'return_pct': (current_balance - initial_balance) / initial_balance * 100
    }


def run_optimization():
    """
    Run grid search optimization over parameter ranges.
    """
    print("=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Fetch data once
    df_entry_raw, df_trend_raw = fetch_data(SYMBOL, HISTORY_LIMIT)
    
    # Calculate Heikin Ashi once (it doesn't change with parameters)
    ha_df = ta.ha(df_entry_raw['open'], df_entry_raw['high'], df_entry_raw['low'], df_entry_raw['close'])
    df_entry_raw = df_entry_raw.join(ha_df)
    
    results = []
    total_combinations = len(PARAM_EMA_FAST) * len(PARAM_EMA_SLOW) * len(PARAM_INDECISION) * len(PARAM_ADX)
    current = 0
    
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    for ema_fast, ema_slow, indecision, adx_thresh in product(PARAM_EMA_FAST, PARAM_EMA_SLOW, PARAM_INDECISION, PARAM_ADX):
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
    print("\n" + "=" * 95)
    print("TOP 10 PARAMETER COMBINATIONS (by Final Portfolio)")
    print("=" * 95)
    print(f"{'Rank':<5} | {'Fast':<5} | {'Slow':<5} | {'Indec.':<8} | {'ADX':<4} | {'Trades':<7} | {'WinRate':<8} | {'Return%':<10} | {'Portfolio':<10}")
    print("-" * 95)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<5} | {r['ema_fast']:<5} | {r['ema_slow']:<5} | {r['indecision']:<8.4f} | {r['adx_threshold']:<4} | {r['total_trades']:<7} | {r['win_rate']:<8.2f}% | {r['return_pct']:<10.2f}% | ${r['final_portfolio']:<10.2f}")
    
    print("-" * 95)
    
    # Print best result
    if results:
        best = results[0]
        print(f"\n🏆 BEST PARAMETERS: EMA_FAST={best['ema_fast']}, EMA_SLOW={best['ema_slow']}, INDECISION={best['indecision']}, ADX={best['adx_threshold']}")
        print(f"   → {best['total_trades']} trades, {best['win_rate']:.2f}% win rate, ${best['final_portfolio']:.2f} final portfolio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heikin Ashi EMA Strategy Backtester')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    args = parser.parse_args()
    
    if args.optimize:
        run_optimization()
    else:
        run_backtest_and_plot()