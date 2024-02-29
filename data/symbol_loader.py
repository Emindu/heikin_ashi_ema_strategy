import json
import requests



exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
response = requests.get(exchange_info_url)
data_json = json.loads(response.text)

#print(data_json["symbols"])

for symbol in data_json["symbols"]:
	if(symbol["quoteAsset"] == "BTC"):
		print(symbol["symbol"])
