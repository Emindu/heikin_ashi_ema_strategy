import pandas as pd
import ccxt

# dataframes = {}
exchange = ccxt.binance()

buy_setups= []
sell_setups = []
increased_volume = []

with open('./data/dataset_usdt.csv') as f:
    symbols = f.read().splitlines()
    del symbols[0]
    for symbol in symbols:
        splitSymbol = symbol.split(',')
        if(splitSymbol[3] == '1'):
            ticker = splitSymbol[1] + "/" + splitSymbol[2]
            bars = exchange.fetch_ohlcv(ticker, timeframe='1h', limit=300)
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            previous_averaged_volume = df['volume'].iloc[1:4:1].mean()
            todays_volume = df['volume'].iloc[-1]
            previous_close = df['close'].iloc[-2]
            current_close = df['close'].iloc[-1]
            if todays_volume > previous_averaged_volume * 4 and previous_close < current_close:
                print(ticker)
                print(previous_averaged_volume)
                print(todays_volume)
                increased_volume.append(ticker)

print(increased_volume)