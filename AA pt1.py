import yfinance as yf
import pandas as pd

#created this dummy dataframe ready to receive the sector performance coefficents from LR
sector_perf = [-2, -1, 0, 1, 2]
#dummy dataframe to test on tickers - will need to be connected to grab list of tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

df = pd.DataFrame(columns=['Ticker', 'Beta', 'Sector_Perf'])

data = []
for ticker, perf in zip(tickers, sector_perf):
    stock = yf.Ticker(ticker)
    beta = stock.info['beta']
    data.append({'Ticker': ticker, 'Beta': beta, 'Sector_Perf': perf})

data = [entry for entry in data if all(pd.notna(list(entry.values())))]

df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

portfolio_pick = pd.DataFrame(columns=['Ticker', 'Beta', 'Sector_Perf'])

portfolio_data = []

for index, row in df.iterrows():
    if row['Sector_Perf'] < 0 and row['Beta'] < 0:
        portfolio_data.append(row)
    elif row['Sector_Perf'] > 0 and row['Beta'] > 0:
        portfolio_data.append(row)

portfolio_pick = pd.concat([portfolio_pick, pd.DataFrame(portfolio_data)], ignore_index=True)