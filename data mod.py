#I think I was doing some data clean up for what probably was a LR between aapl and sp500 returns

import pandas as pd
import yfinance as yf

aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-01-01', progress=False)

ffunds_path = [r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\FEDFUNDS.csv']
ffunds_path_str = ffunds_path[0]
ffunds = pd.read_csv(ffunds_path_str)

ffunds['DATE'] = pd.to_datetime(ffunds['DATE'])
ffunds.set_index('DATE', inplace=True)
ffunds.index = ffunds.index.to_period('M').to_timestamp('M')

monthly_aapl = aapl.resample('ME').mean()
monthly_sp500 = sp500.resample('ME').mean()

monthly_aapl['Return'] = (monthly_aapl['Close'].shift(-1) - monthly_aapl['Close']) / monthly_aapl['Close']
monthly_sp500['Return'] = (monthly_sp500['Close'].shift(-1) - monthly_sp500['Close']) / monthly_sp500['Close']
#working site

#print(aapl[['Close', 'Return']].head(25))
#print(sp500[['Close', 'Return']].head(25))

data = pd.DataFrame(index=monthly_aapl.index)
data['AAPL_Return'] = monthly_aapl['Return']
data['SP500_Return'] = monthly_sp500['Return']
data['ffunds'] = ffunds['FEDFUNDS']
data['Target'] = (data['AAPL_Return'] > data['SP500_Return']).astype(int)

#with pd.option_context('display.max_rows', None):
    #print(ffunds)

