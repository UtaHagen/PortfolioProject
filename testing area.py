import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score, classification_report

#importing the data
ffunds_path = [r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\FEDFUNDS.csv']
ffunds_path_str = ffunds_path[0]

#reading the files
aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-01-01', progress=False)
ffunds = pd.read_csv(ffunds_path_str)

#ffunds Data Cleanup + Manipulation
ffunds['DATE'] = pd.to_datetime(ffunds['DATE'])
ffunds.set_index('DATE', inplace=True)

#aapl + sp500 Data Manipulation
aapl['Return'] = (aapl['Close'].shift(-21) - aapl['Close']) / aapl['Close']
sp500['Return'] = (sp500['Close'].shift(-21) - sp500['Close']) / sp500['Close']

aapl['monthly_average'] = aapl['Return'].resample('ME').mean()
sp500['monthly_average'] = sp500['Return'].resample('ME').mean()

data = pd.DataFrame(index=aapl.index)
data['AAPL_MA'] = aapl['monthly_average']
data['SP500_MA'] = sp500['monthly_average']
data['Target'] = (data['AAPL_MA'] > data['SP500_MA']).astype(int)

data.dropna(inplace=True)

#print(data)

def dataframe_length(df):
    length = len(df)
    print(f"The DataFrame has {length} rows.")

def check_nan(df):
    nan_df = df.isnull()
    print("Number of NaN values in each column:\n", nan_df.sum())

with pd.option_context('display.max_rows', None):
    print(aapl['Return'])

# To use this function, pass your DataFrame to it like this:
# dataframe_length(your_dataframe)

#dataframe_length(aapl)
#dataframe_length(data)
#dataframe_length(ffunds)

#data = pd.DataFrame(index=aapl.index)
#data['AAPL_Return'] = aapl['monthly_average']
#data['SP500_Return'] = sp500['monthly_average']
#data['Fed Rate'] = ffunds['FEDFUNDS']

#print(data)