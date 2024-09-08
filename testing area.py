import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

portfolio_pick = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\portfolio_pick.csv'
portfolio_pick_df = pd.read_csv(portfolio_pick)

def is_last_day_of_month(date):
    next_day = date + pd.DateOffset(days=1)
    return next_day.month != date.month

def calculate_recent_monthly_return(ticker):
    try:
        stock_data = yf.download(ticker, period='1y', interval='1d')
        stock_data = stock_data.resample('ME').last()
        stock_data['Monthly Return'] = stock_data['Adj Close'].pct_change()
        stock_data['Ticker'] = ticker
        
        today = pd.Timestamp(datetime.now().date())
        if is_last_day_of_month(today):
            return stock_data[['Ticker', 'Monthly Return']].iloc[-1]
        else:
            return stock_data[['Ticker', 'Monthly Return']].iloc[-2]
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        return pd.Series({'Ticker': ticker, 'Monthly Return': None})

results = []
for ticker in portfolio_pick_df['Stock']:
    result = calculate_recent_monthly_return(ticker)
    results.append(result)

monthly_returns_df = pd.DataFrame(results)
monthly_returns_df.to_csv(r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\monthly_returns.csv', index=False)

portfolio_pick = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\portfolio_pick.csv'
portfolio_pick_df = pd.read_csv(portfolio_pick)

def is_last_day_of_month(date):
    next_day = date + pd.DateOffset(days=1)
    return next_day.month != date.month

def calculate_recent_monthly_return(ticker):
    try:
        stock_data = yf.download(ticker, period='1y', interval='1d')
        stock_data = stock_data.resample('ME').last()
        stock_data['Monthly Return'] = stock_data['Adj Close'].pct_change()
        stock_data['Ticker'] = ticker
        
        today = pd.Timestamp(datetime.now().date())
        if is_last_day_of_month(today):
            return stock_data[['Ticker', 'Monthly Return']].iloc[-1]
        else:
            return stock_data[['Ticker', 'Monthly Return']].iloc[-2]
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        return pd.Series({'Ticker': ticker, 'Monthly Return': None})

results = []
for ticker in portfolio_pick_df['Stock']:
    result = calculate_recent_monthly_return(ticker)
    results.append(result)

monthly_returns_df = pd.DataFrame(results)
#monthly_returns_df.to_csv(r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\monthly_returns.csv', index=False)