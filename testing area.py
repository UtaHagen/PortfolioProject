import pandas as pd
import yfinance as yf
import pandas_ta as ta
import statsmodels.api as sm
from itertools import chain, combinations

#Data port + manipulation (aapl)
aaplWR = yf.download('AAPL', start='2008-12-31', end='2024-01-01', progress=False)
aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
monthly_aaplWR = aaplWR.resample('ME').mean()
monthly_aapl = aapl.resample('ME').mean()

#Calculating RSI
aaplRSI = pd.DataFrame()
aaplRSI = aapl.ta.rsi(append=True)

print(aaplRSI)