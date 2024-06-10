#!/usr/bin/env python
# coding: utf-8

# In[12]:


import yfinance as yf
import pandas as pd

#S&P 500 tickers
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
sp500_table = tables[0]
sp500_tickers = sp500_table['Symbol'].tolist()

# Risk-Free Rate: 10yr treasury note
tnx = yf.Ticker('^TNX')
tnx_info = tnx.info
rfr = tnx_info.get('previousClose') / 100

# Equity risk premium source: Kroll
erp = 0.055

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        beta = stock.info.get('beta')
        dividend_yield = stock.info.get('dividendYield')
        dividend_rate = stock.info.get('dividendRate')
        roe = stock.info.get('returnOnEquity')
        payout_ratio = stock.info.get('payoutRatio')

        if beta is None or dividend_yield is None or dividend_rate is None or roe is None or payout_ratio is None:
            return None
#gordon growth model DDM
#NOTE intrinsic values are very high...calcualtions seem correct...maybe take this out
        coe = rfr + beta * erp
        retention_ratio = 1 - payout_ratio
        dividend_growth_rate = roe * retention_ratio
        dividend_growth_rate = min(dividend_growth_rate, coe * 0.9)  # .9 to prevent growth rate being higher than what investors expect.. sustainability. other values could be 1 or .8. depends on what we want

        if dividend_rate and coe > dividend_growth_rate:
            intrinsic_value = dividend_rate / (coe - dividend_growth_rate)
            return {
                'Ticker': ticker,
                'Dividend_Yield': dividend_yield,
                'Dividend_Rate': dividend_rate,
                'Intrinsic_Value': intrinsic_value,
                'Beta': beta
            }
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
    return None

# Get data for all S&P 500 stocks
stock_data = []
for ticker in sp500_tickers:
    data = fetch_stock_data(ticker)
    if data:
        stock_data.append(data)

# Dataframe of results
df = pd.DataFrame(stock_data)

#Weights
df['Score'] = (df['Dividend_Yield'] * 0.4 + df['Dividend_Rate'] * 0.3 + df['Intrinsic_Value'] * 0.3)

# Sort by score
df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Top ten
print(df.head(10))


# In[ ]:




