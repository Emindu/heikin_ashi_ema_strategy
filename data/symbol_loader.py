import json
import requests
import csv
import sys
import time


exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
response = requests.get(exchange_info_url)
data_json = json.loads(response.text)

#print(data_json["symbols"])
symbols = []
print(sys.argv[1])
for symbol in data_json["symbols"]:
	if(symbol["quoteAsset"] == sys.argv[1]):
		symbolTicker = symbol["symbol"]
		time.sleep(2)
		validity_check_url = "https://www.binance.com/en/trade/" + symbolTicker
		validity = 0
		response_validity = requests.get(validity_check_url)
		print(response_validity.status_code)
		if(response_validity.status_code == 200 ):
			validity = 1
		quoteAsset = symbol["quoteAsset"]
		baseAsset = symbol["baseAsset"]
		symbol_data = [symbolTicker, baseAsset , quoteAsset, validity]
		symbols.append(symbol_data)


print(len(symbols))

with open('dataset_' +sys.argv[1]+ '.csv', 'w', newline='') as file:
	writer = csv.writer(file, delimiter=',' ,quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['ticker','symbol','base_symbol','active'])
	for i in symbols:
		writer.writerow(i)