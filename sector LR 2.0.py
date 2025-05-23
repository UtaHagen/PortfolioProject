#This generates the results.csv file with all of the sector and their coefficients for future performance?

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import statsmodels.api as sm

sp500_df = yf.download('^GSPC', start='2010-01-01')
monthly_sp500 = sp500_df.resample('ME').mean()
monthly_sp500['Return'] = (monthly_sp500['Close'].shift(-21) - monthly_sp500['Close']) / monthly_sp500['Close']

sectors_path = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\sectorsxd - Sheet1.csv'
sectors_df = pd.read_csv(sectors_path)

results_df = pd.DataFrame(columns=['Sector', 'Coefficient'])

for sector_symbol in sectors_df['Sector']:
    print(f'Processing sector: {sector_symbol}')
    
    sector_df = yf.download(sector_symbol, start='2010-01-01')
    monthly_sector = sector_df.resample('M').mean()
    monthly_sector['Return'] = (monthly_sector['Close'].shift(-21) - monthly_sector['Close']) / monthly_sector['Close']

    monthly_sp500.dropna(inplace=True)
    monthly_sector.dropna(inplace=True)

    returns_df = pd.DataFrame({
        'SP500_Return': monthly_sp500['Return'],
        'Sector_Return': monthly_sector['Return']
    }).dropna()

    X = returns_df['SP500_Return'].values
    y = returns_df['Sector_Return'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_sm = sm.add_constant(X_train)  
    model = sm.OLS(y_train, X_train_sm).fit()

    y_pred = model.predict(sm.add_constant(X_test))

    new_row = pd.DataFrame({
        'Sector': [sector_symbol],
        'Coefficient': [model.params[1]]
    })

    results_df = pd.concat([results_df, new_row], ignore_index=True)

    print(f'Sector: {sector_symbol}')
    print(f'Coefficient: {model.params[1]}')

results_df.to_csv(r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\results.csv', index=False)