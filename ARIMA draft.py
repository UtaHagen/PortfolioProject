import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np

tickerSymbols = ['AAPL', '^GSPC']

for tickerSymbol in tickerSymbols:
    tickerData = yf.Ticker(tickerSymbol)

    tickerDf = tickerData.history(period='10y')

    tickerDf['Pct_Change'] = tickerDf['Close'].pct_change() * 100

    monthly_average = tickerDf['Pct_Change'].resample('ME').mean()

#cleaning monthly_average to address DateTime
monthly_average = monthly_average.reset_index(drop = True)

#examining the data
monthly_average.plot(label = tickerSymbol)

msk = (monthly_average.index < len(monthly_average)-30)
df_train = monthly_average[msk].copy()
df_test = monthly_average[~msk].copy()

#Checking for stationarity of time series

#Method 1 - ACF and PACF plot
acf_original = plot_acf(df_train)

pacf_original = plot_pacf(df_train)

#plt.show()

#Method 2 - ADF Test

adf_test = adfuller(df_train)
#print(f'p-value: {adf_test[1]}')