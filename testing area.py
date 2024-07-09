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

#Data Cleanup + Manipulation

##Federal Funds Rate Data Manipulation
ffunds['DATE'] = pd.to_datetime(ffunds['DATE'])
ffunds.set_index('DATE', inplace=True)

#aapl + sp500 Data Manipulation

# Calculate daily returns
aapl['Return'] = aapl['Close'].pct_change()

# Fill NaN values with 0
aapl['Return'] = aapl['Return'].fillna(0)

# Calculate cumulative returns
cumulative_returns = aapl['Return'].add(1).cumprod()

# Resample to get the last cumulative return of each month
end_of_month_cumulative_returns = cumulative_returns.resample('M').last()

# Get monthly returns as change in cumulative monthly returns
aapl['monthly_return'] = end_of_month_cumulative_returns.pct_change()

# Calculate 'MR' as the change in 'monthly_return' over a 21-day period
#aapl['MR'] = (aapl['monthly_return'].shift(-21) - aapl['monthly_return']) / aapl['monthly_return']


print(aapl['Return'])
print(aapl['monthly_return'])
#print(aapl['MR'])
print(cumulative_returns)

#data = pd.DataFrame(index=aapl.index)
#data['AAPL_MA'] = aapl['monthly_average']
#data['SP500_MA'] = sp500['monthly_average']
#data['Target'] = (data['AAPL_MA'] > data['SP500_MA']).astype(int)

#data.dropna(inplace=True)

#print(data)

def dataframe_length(df):
    length = len(df)
    print(f"The DataFrame has {length} rows.")

def check_nan(df):
    nan_df = df.isnull()
    print("Number of NaN values in each column:\n", nan_df.sum())

#with pd.option_context('display.max_rows', None):
#    print(aapl['Return'])

# To use this function, pass your DataFrame to it like this:
# dataframe_length(your_dataframe)

#dataframe_length(aapl['MR'])
#dataframe_length(sp500['MR'])
#dataframe_length(ffunds)

#data = pd.DataFrame(index=aapl.index)
#data['AAPL_Return'] = aapl['monthly_average']
#data['SP500_Return'] = sp500['monthly_average']
#data['Fed Rate'] = ffunds['FEDFUNDS']

#print(data)