import os, pandas
import ccxt
import sys
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import sys
import datetime
import csv

# dataframes = {}
exchange = ccxt.binance()

buy_setups= []
sell_setups = []


with open('./data/dataset_usdt.csv') as f:
    symbols = f.read().splitlines()
    del symbols[0]
    for symbol in symbols:
        splitSymbol = symbol.split(',')
        if(splitSymbol[3] == '1'):
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

                current_open_price = 0
                try:
                    current_open_price = df.iloc[-1]['open']
                except:
                    print("error on opening price")

                def find_green_indecision(df):
                    if find_undecision_bar(df):
                        if (df.iloc[-3]['HA_close'] > df.iloc[-3]['HA_open']):
                            return True
                        else:
                            return False
        
            
                def find_red_undecission(df):
                    if find_undecision_bar(df):
                        if (df.iloc[-3]['HA_close'] < df.iloc[-3]['HA_open']):
                            return True
                        else:
                            return False
        

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
                    if(find_green_indecision(df)):
                        # print('{} close {} low {} high {} open'.format(df.iloc[-3]['HA_close'], df.iloc[-3]['HA_low'] , df.iloc[-3]['HA_high'] ,df.iloc[-3]['HA_open']))
                        if (df.iloc[-2]['HA_close'] > df.iloc[-2]['HA_low'] and df.iloc[-2]['HA_high'] == df.iloc[-2]['HA_open']):
                            print("{} sell signal bar ditected ".format(symbol)  )
                            url = "https://www.binance.com/en/trade/{}".format(splitSymbol[0])
                            trading_view_url= "https://www.tradingview.com/chart/OfW7LbeC/?symbol=BINANCE%3A" + splitSymbol[0]
                            print(url)
                            try:
                                symbol_data = [splitSymbol[0], url , current_open_price, "sell", "", "  ", trading_view_url]
                                sell_setups.append(symbol_data)
                            except Exception as e:
                                print(e)
                        


                def find_buy_signal(df):
                    if(find_red_undecission(df)):
                        # print('{} close {} low {} high {} open'.format(df.iloc[-3]['HA_close'], df.iloc[-3]['HA_low'] , df.iloc[-3]['HA_high'] ,df.iloc[-3]['HA_open']))
                        if (df.iloc[-2]['HA_close'] > df.iloc[-2]['HA_open'] and df.iloc[-2]['HA_open'] == df.iloc[-2]['HA_low']):
                            print("{} buy signal bar ditected ".format(symbol)  )
                            url = "https://www.binance.com/en/trade/{}".format(splitSymbol[0])
                            trading_view_url= "https://www.tradingview.com/chart/OfW7LbeC/?symbol=BINANCE%3A" + splitSymbol[0]

                            print(url)
                            try: 
                                symbol_data = [splitSymbol[0], url , current_open_price, "buy", "", "  ", trading_view_url]
                                buy_setups.append(symbol_data)
                            except Exception as e:
                                print(e)

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
filebase = f"./signals/{month:02d}-{day:02d}_{hour:02d}:{minutes:02d}.csv"
    

header = ['ticker','url','entry_price', 'side', 'price_after_1h', 'price_after_4h', 'tradingview_url']
with open(filebase, 'w', newline='') as file:
	writer = csv.writer(file, delimiter=',' ,quoting=csv.QUOTE_MINIMAL)
	writer.writerow(header)
	for i in buy_setups:
		writer.writerow(i)          
	for i in sell_setups:
		writer.writerow(i)      