import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=IBM&apikey=6WLY2DVR35CPCC7A'
r = requests.get(url)
data = r.json()

print(data)