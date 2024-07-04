import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np

csv_files = [r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\FEDFUNDS.csv', 
             r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\M1SL.csv', 
             r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\MORTGAGE15US.csv']

dataframes = [pd.read_csv(file) for file in csv_files]

ffunds = dataframes[0]['FEDFUNDS']
m1sl = dataframes[1]['M1SL_CHG']
mtg15 = dataframes[2]['MORTGAGE15US']

df = pd.concat([ffunds,m1sl,mtg15], axis=1, keys=['FEDFUNDS','M1SL_CHG','MORTGAGE15US'])

tickerSymbols = ['AAPL', '^GSPC']

for tickerSymbol in tickerSymbols:
    tickerData = yf.Ticker(tickerSymbol)

    tickerDf = tickerData.history(period='10y')

    tickerDf['Pct_Change'] = tickerDf['Close'].pct_change() * 100

    monthly_average = tickerDf['Pct_Change'].resample('ME').mean()

#code for the multi-linear regression model

X = df[['FEDFUNDS','M1SL_CHG', 'MORTGAGE15US']]
y = monthly_average.reset_index(drop = True)

X = sm.add_constant(X)

model = sm.OLS(y,X).fit()

print(model.summary())