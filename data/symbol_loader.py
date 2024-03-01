import json
import requests
import csv
import sys


exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
response = requests.get(exchange_info_url)
data_json = json.loads(response.text)

#print(data_json["symbols"])
symbols = []
print(sys.argv[1])
for symbol in data_json["symbols"]:
	if(symbol["quoteAsset"] == sys.argv[1]):
		symbolTicker = symbol["symbol"]
		quoteAsset = symbol["quoteAsset"]
		baseAsset = symbol["baseAsset"]
		symbol_data = [symbolTicker, baseAsset , quoteAsset]
		symbols.append(symbol_data)


print(len(symbols))

with open('dataset_' +sys.argv[1]+ '.csv', 'w', newline='') as file:
	writer = csv.writer(file, delimiter=',' ,quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['ticker','symbol','base_symbol'])
	for i in symbols:
		writer.writerow(i)