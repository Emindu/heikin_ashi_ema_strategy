import os, pandas
import ccxt
import sys
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import sys
import datetime

# dataframes = {}
exchange = ccxt.binance()

buy_signals = []
sell_signals = []


with open('./data/dataset_USDT.csv') as f:
    symbols = f.read().splitlines()
    del symbols[0]
    for symbol in symbols:
        splitSymbol = symbol.split(',')
        ticker = splitSymbol[1] + "/" + splitSymbol[2]
        try:
            bars = exchange.fetch_ohlcv(ticker, timeframe='1h', limit=100)
            bars4 = exchange.fetch_ohlcv(ticker, timeframe='4h', limit=100)
            # print(ticker)
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df4 = pd.DataFrame(bars4, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df4['time'] = pd.to_datetime(df4['time'], unit='ms')
            df4.set_index('time', inplace=True, drop=True)



            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True, drop=True)
            # df['20sma'] = df['close'].rolling(window=20).mean()


            #HA candle data
            ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])



            #find sma 10 ema 30
            df4['30ema'] = df4['close'].rolling(window=30).mean()
            df4['10ema'] = df4['close'].rolling(window=10).mean()
            # if 30ema > 10 ema find sell signals 1 hour


            def find_undecision_bar(df):
                # must be positive below two
                if(df.iloc[-3]['HA_high'] > df.iloc[-3]['HA_close'] and df.iloc[-3]['HA_low'] < df.iloc[-3]['HA_open']):
                    high_open_diff = round((df.iloc[-3]['HA_high'] - df.iloc[-3]['HA_open'])/df.iloc[-3]['HA_high'], 3)
                    low_close_diff = round((df.iloc[-3]['HA_close'] - df.iloc[-3]['HA_low'])/df.iloc[-3]['HA_close'], 3 )
                    if (high_open_diff > 0.005 and low_close_diff > 0.005 ):
                        return True
                    else:
                        return False
                else:
                    return False

            def find_sell_signal(df):
                if(find_undecision_bar(df)):
                    # print('{} close {} low {} high {} open'.format(df.iloc[-3]['HA_close'], df.iloc[-3]['HA_low'] , df.iloc[-3]['HA_high'] ,df.iloc[-3]['HA_open']))
                    if (df.iloc[-2]['HA_close'] > df.iloc[-2]['HA_low'] and df.iloc[-2]['HA_high'] == df.iloc[-2]['HA_open']):
                        print("{} sell signal bar ditected ".format(symbol)  )
                        sell_signals.append("https://www.binance.com/en/trade/{}".format(splitSymbol[0]))
                        


            def find_buy_signal(df):
                if(find_undecision_bar(df)):
                    # print('{} close {} low {} high {} open'.format(df.iloc[-3]['HA_close'], df.iloc[-3]['HA_low'] , df.iloc[-3]['HA_high'] ,df.iloc[-3]['HA_open']))
                    if (df.iloc[-2]['HA_close'] > df.iloc[-2]['HA_open'] and df.iloc[-2]['HA_open'] == df.iloc[-2]['HA_low']):
                        print("{} buy signal bar ditected ".format(symbol)  )
                        buy_signals.append("https://www.binance.com/en/trade/{}".format(splitSymbol[0]))


            #if 30ema > 10 ema find sell signlas
            if (df4.iloc[-2]['30ema'] > df4.iloc[-2]['10ema']):
                find_sell_signal(ha_df)


            #if 10ema > 30 ema find buy signal on 1 hour
            if (df4.iloc[-2]['10ema'] > df4.iloc[-2]['30ema']):
                find_buy_signal(ha_df)


        except:
            print("Symbol error", ticker)
            print("Oops!", sys.exc_info()[0], "occurred.")
            continue
        # print(ticker)

current_time = datetime.datetime.now()


# Extract hour and minutes
hour = current_time.hour
minutes = current_time.minute
day = current_time.day
month = current_time.month

def saveSignals(setup, side):
    filebase = f"./signals/{month:02d}-{day:02d}_{hour:02d}:{minutes:02d}.txt"
    with open(filebase, "w") as file:
        file.write(side + "\n")
        for line in setup:
            file.write(line + "\n")
                
print("buy signals")
print(*buy_signals, sep = "\n")
saveSignals(buy_signals, "buy setups")

print("sell signals")
print(*sell_signals, sep = "\n")
saveSignals(sell_signals, "sell setups")
