#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def williams_r(data, lookback=14):
    highh = data['High'].rolling(lookback).max()
    lowl = data['Low'].rolling(lookback).min()
    wr = -100 * ((highh - data['Close']) / (highh - lowl))
    return wr

#stock picks from Xialin's model
csv_path = '/Users/allison/Desktop/pick.csv'
#fedfunds
ffunds_path = '/Users/allison/Desktop/FEDFUNDS.csv'
#s&p 500 and associated sector ETFs
sector_path = '/Users/allison/Desktop/constituentssp.csv'

stocks_df = pd.read_csv(csv_path)
ffunds = pd.read_csv(ffunds_path)
ffunds['DATE'] = pd.to_datetime(ffunds['DATE'])
ffunds.set_index('DATE', inplace=True)
sectors_df = pd.read_csv(sector_path)

print("Sector CSV columns:", sectors_df.columns)
print(sectors_df.head())

stock_to_sector = dict(zip(sectors_df['Stock'], sectors_df['Sector']))

results = []

for index, row in stocks_df.iterrows():
    company = row[0]
  
    if company not in stock_to_sector:
        print(f"No sector found for {company}")
        continue
    
    sector_etf = stock_to_sector[company]

    print(f"Company: {company}, Sector ETF: {sector_etf}")

    company_data = yf.download(company, start='2010-01-01', end='2024-01-01', progress=False)
    sector_data = yf.download(sector_etf, start='2010-01-01', end='2024-01-01', progress=False)

    print(f"Company data head: {company_data.head()}")
    print(f"Sector ({sector_etf}) data head: {sector_data.head()}")

    if company_data.empty or sector_data.empty:
        print(f"Data for {company} or {sector_etf} is empty")
        continue

    monthly_company = company_data.resample('M').mean()
    monthly_sector = sector_data.resample('M').mean()

    if 'Close' not in monthly_company.columns or 'Close' not in monthly_sector.columns:
        print(f"Close column missing in data for {company} or {sector_etf}")
        continue

    monthly_company['Return'] = (monthly_company['Close'].shift(-21) - monthly_company['Close']) / monthly_company['Close']
    monthly_sector['Return'] = (monthly_sector['Close'].shift(-21) - monthly_sector['Close']) / monthly_sector['Close']

    company_data['Williams_%R'] = williams_r(company_data)
    monthly_company['Williams_%R'] = company_data['Williams_%R'].resample('M').last()

    data = pd.DataFrame(index=monthly_company.index)
    data['Company_Return'] = monthly_company['Return']
    data['Sector_Return'] = monthly_sector['Return']
    data['Williams_%R'] = monthly_company['Williams_%R']

    monthly_ffunds = ffunds.resample('M').mean()
    data = data.join(monthly_ffunds['FEDFUNDS'])

    data['Target'] = (data['Company_Return'] > data['Sector_Return']).astype(int)

    data.dropna(inplace=True)

    X = data[['Company_Return', 'Sector_Return', 'FEDFUNDS', 'Williams_%R']]
    y = data['Target']

    random_state = np.random.randint(0, 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Company: {company}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    new_data = X_test.iloc[-1:].copy()
    prediction = model.predict(new_data)
    prediction_text = f"{company} will outperform {sector_etf}" if prediction[0] == 1 else f"{company} will underperform {sector_etf}"
    print("Prediction: ", prediction_text)
    
    results.append([company, sector_etf, accuracy, prediction_text])
    
    print("\n" + "="*50 + "\n")

results_df = pd.DataFrame(results, columns=['Company', 'Sector ETF', 'Accuracy', 'Prediction'])
print(results_df)


# In[28]:


outperform_results = results_df[results_df['Prediction'].str.contains("outperform")]

print(outperform_results)


# In[ ]:




