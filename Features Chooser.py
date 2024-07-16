import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from itertools import chain, combinations

#Data port + manipulation (aapl)
aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
monthly_aapl = aapl.resample('ME').mean()

#Federal Funds Port + Manipulation
ffunds_path = [r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\FEDFUNDS.csv']
ffunds_path_str = ffunds_path[0]
ffunds = pd.read_csv(ffunds_path_str)

ffunds = ffunds.drop(ffunds.index[0])
ffunds['DATE'] = pd.to_datetime(ffunds['DATE'])
ffunds.set_index('DATE', inplace=True)
ffunds.index = ffunds.index.to_period('M').to_timestamp('M')

#William %R - Data manipulation
high14 = monthly_aapl['High'].rolling(14).max()
low14 = monthly_aapl['Low'].rolling(14).min()
monthly_aapl['Williams %R'] = -100 * ((high14 - monthly_aapl['Close']) / (high14 - low14))
monthly_aapl['Return'] = monthly_aapl['Close'].pct_change()

#Combining Data
data = pd.DataFrame(index=aapl.index)
data['Williams %R'] = monthly_aapl['Williams %R']
data['ffunds'] = ffunds['FEDFUNDS']

data.dropna(inplace=True)

y = monthly_aapl['Return']
X = data[['Williams %R','ffunds']]

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


results = []
for combo in powerset(X.columns):
    combo = list(combo)
    X_subset = X[combo]
    
    X_subset = sm.add_constant(X_subset)
    model = sm.OLS(y, X_subset)
    results.append((combo, model.fit().rsquared))


results.sort(key=lambda x: x[1], reverse=True)
top_10 = results[:10]

for combo, rsquared in top_10:
    print(f"Combo: {combo}, R-squared: {rsquared}")