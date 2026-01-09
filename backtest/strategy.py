# backtest/strategy.py
"""Core trading strategy and backtest logic"""

import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import (
    SYMBOL, HISTORY_LIMIT, INDECISION_THRESHOLD, ADX_THRESHOLD,
    LEVERAGE, TRADING_FEE, RISK_PER_TRADE, STOP_LOSS_ATR_MULT
)
from .data import fetch_data
from .indicators import prepare_indicators, find_undecision_bar


def run_backtest_and_plot(ema_fast: int = 10, ema_slow: int = 30):
    """
    Run backtest with visualization and detailed trade logging.
    """
    # Fetch and prepare data
    df_entry, df_trend = fetch_data(SYMBOL, HISTORY_LIMIT)
    
    # Remove duplicates and sort
    df_entry = df_entry[~df_entry.index.duplicated(keep='first')].sort_index()
    df_trend = df_trend[~df_trend.index.duplicated(keep='first')].sort_index()
    
    # Prepare indicators
    df_entry, df_trend = prepare_indicators(df_entry, df_trend, ema_fast, ema_slow)
    
    # Initialize tracking variables
    trades = []
    position = None 
    entry_price = 0
    stop_loss = 0
    position_size = 0
    
    # Signal columns for visualization
    df_entry['buy_signal'] = float('nan')
    df_entry['sell_signal'] = float('nan')
    df_entry['buy_signal_plot'] = float('nan')
    df_entry['sell_signal_plot'] = float('nan')
    df_entry['exit_signal'] = float('nan')
    
    # Align 4H indicators to 1H timeframe
    df_aligned = df_trend.reindex(df_entry.index, method='ffill')
    df_entry['trend_fast_ema'] = df_aligned['fast_ema']
    df_entry['trend_slow_ema'] = df_aligned['slow_ema']
    df_entry['trend_adx'] = df_aligned['adx']

    # Calculate ATR for stop-loss and plotting
    df_entry['atr'] = ta.atr(df_entry['high'], df_entry['low'], df_entry['close'], length=14)
    df_entry['atr'] = df_entry['atr'].bfill()
    
    ATR_MULTIPLIER = 1.0

    print("Running simulation...")
    
    for i in range(30, len(df_entry) - 1):
        current_bar = df_entry.iloc[i]
        prev_bar = df_entry.iloc[i-1]
        current_time = df_entry.index[i]
        
        ema_fast_val = current_bar['trend_fast_ema']
        ema_slow_val = current_bar['trend_slow_ema']
        adx_value = current_bar['trend_adx']
        
        if pd.isna(ema_fast_val) or pd.isna(ema_slow_val):
            continue
        current_trend = 1 if ema_fast_val > ema_slow_val else -1
        
        adx_ok = (not pd.isna(adx_value)) and (adx_value >= ADX_THRESHOLD)

        # ENTRY LOGIC
        if position is None:
            is_indecision = find_undecision_bar(prev_bar, INDECISION_THRESHOLD)
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
                    
                    plot_y = df_entry.iloc[i]['low'] - (atr_value * ATR_MULTIPLIER)
                    df_entry.loc[df_entry.index[i+1], 'buy_signal'] = entry_price
                    df_entry.loc[df_entry.index[i+1], 'buy_signal_plot'] = plot_y

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
                    
                    plot_y = df_entry.iloc[i]['high'] + (atr_value * ATR_MULTIPLIER)
                    df_entry.loc[df_entry.index[i+1], 'sell_signal'] = entry_price
                    df_entry.loc[df_entry.index[i+1], 'sell_signal_plot'] = plot_y

        # EXIT LOGIC
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
                
                trades.append({
                    'type': 'SHORT' + (' (SL)' if stopped_out else ''),
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': portfolio_pnl,
                    'time': df_entry.index[i+1] if not stopped_out else df_entry.index[i]
                })
                df_entry.loc[df_entry.index[i+1], 'exit_signal'] = exit_price
                position = None

    # Calculate and print metrics
    _print_metrics(trades)
    
    # Generate plot
    _generate_plot(df_entry, ema_fast, ema_slow)


def run_backtest(df_entry: pd.DataFrame, df_trend: pd.DataFrame, 
                 ema_fast: int, ema_slow: int, 
                 indecision_threshold: float, adx_threshold: float = 25) -> dict:
    """
    Run backtest with specific parameters and return metrics (no plotting).
    """
    df_entry = df_entry.copy(deep=True)
    df_trend = df_trend.copy(deep=True)
    
    df_entry = df_entry[~df_entry.index.duplicated(keep='first')].sort_index()
    df_trend = df_trend[~df_trend.index.duplicated(keep='first')].sort_index()
    
    df_entry, df_trend = prepare_indicators(df_entry, df_trend, ema_fast, ema_slow)
    
    # Calculate ATR
    df_entry['atr'] = ta.atr(df_entry['high'], df_entry['low'], df_entry['close'], length=14)
    df_entry['atr'] = df_entry['atr'].bfill()
    
    trades = []
    position = None 
    entry_price = 0
    stop_loss = 0
    position_size = 0
    
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


def _print_metrics(trades: list):
    """Print trade details and summary metrics."""
    initial_balance = 100
    current_balance = initial_balance
    winning_trades = 0

    print("\n" + "="*70)
    print(f"{'Type':<10} | {'Date':<20} | {'Entry':<10} | {'Exit':<10} | {'PnL (%)':<8}")
    print("-" * 70)

    for trade in trades:
        pnl = trade['pnl']
        if pnl > 0:
            winning_trades += 1
        
        print(f"{trade['type']:<10} | {str(trade['time']):<20} | {trade['entry']:<10.4f} | {trade['exit']:<10.4f} | {pnl:<8.2f}%")
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


def _generate_plot(df_entry: pd.DataFrame, ema_fast: int, ema_slow: int):
    """Generate interactive Plotly chart."""
    print("Generating Plot...")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Heikin Ashi Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_entry.index,
        open=df_entry['HA_open'],
        high=df_entry['HA_high'],
        low=df_entry['HA_low'],
        close=df_entry['HA_close'],
        name='Heikin Ashi'
    ), row=1, col=1)

    # Buy signals
    buy_signals = df_entry.dropna(subset=['buy_signal_plot'])
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['buy_signal_plot'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='lime'),
            name='Buy Signal'
        ), row=1, col=1)

    # Sell signals
    sell_signals = df_entry.dropna(subset=['sell_signal_plot'])
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['sell_signal_plot'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Sell Signal'
        ), row=1, col=1)

    # Exit signals
    exit_signals = df_entry.dropna(subset=['exit_signal'])
    if not exit_signals.empty:
        fig.add_trace(go.Scatter(
            x=exit_signals.index,
            y=exit_signals['exit_signal'],
            mode='markers',
            marker=dict(symbol='x', size=10, color='white'),
            name='Exit'
        ), row=1, col=1)

    # Trend EMAs
    fig.add_trace(go.Scatter(
        x=df_entry.index,
        y=df_entry['trend_fast_ema'],
        mode='lines',
        line=dict(color='cyan', width=1),
        name=f'Fast EMA ({ema_fast})'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_entry.index,
        y=df_entry['trend_slow_ema'],
        mode='lines',
        line=dict(color='orange', width=1),
        name=f'Slow EMA ({ema_slow})'
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=df_entry.index,
        y=df_entry['volume'],
        name='Volume',
        marker_color='rgba(100, 100, 100, 0.5)'
    ), row=2, col=1)

    fig.update_layout(
        title=f'{SYMBOL} - Heikin Ashi EMA Strategy Backtest',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=800
    )

    fig.show()
