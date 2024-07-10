import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from itertools import chain, combinations

#Data port + manipulation
aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
monthly_aapl = aapl.resample('ME').mean()

high14 = monthly_aapl['High'].rolling(14).max()
low14 = monthly_aapl['Low'].rolling(14).min()
monthly_aapl['Williams %R'] = -100 * ((high14 - monthly_aapl['Close']) / (high14 - low14))
monthly_aapl['Return'] = monthly_aapl['Close'].pct_change()

monthly_aapl.dropna(inplace=True)

y = monthly_aapl['Return']
X = monthly_aapl.drop('Return', axis=1)

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