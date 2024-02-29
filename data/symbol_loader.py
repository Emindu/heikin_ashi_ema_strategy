import json
import requests
import csv


exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
response = requests.get(exchange_info_url)
data_json = json.loads(response.text)

#print(data_json["symbols"])
symbols = []

for symbol in data_json["symbols"]:
	if(symbol["quoteAsset"] == "BTC"):
		a = symbol["symbol"]
		symbol_data = [a,a[:-3],a[-3:]]
		symbols.append(symbol_data)


print(len(symbols))

with open('dataset_btc.csv', 'w', newline='') as file:
	writer = csv.writer(file, delimiter=',' ,quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['ticker','symbol','base_symbol'])
	for i in symbols:
		writer.writerow(i)