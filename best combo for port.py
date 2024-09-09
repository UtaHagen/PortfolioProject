from itertools import combinations
from fractions import Fraction
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

monthly_returns_path = r'Data\portfolio_pick.csv'
monthly_returns = pd.read_csv(monthly_returns_path)

def calculate_most_recent_monthly_return(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2mo")
    hist['Monthly Return'] = hist['Close'].pct_change(periods=21)
    hist = hist[['Monthly Return']].dropna()

    today = datetime.today()
    if today.day == 1:
        most_recent_return = hist.iloc[-1]
    else:
        most_recent_return = hist.iloc[-2]

    return most_recent_return['Monthly Return']

stock_dataframes = {}

for ticker in monthly_returns['Stock']:
    stock_dataframes[ticker] = calculate_most_recent_monthly_return(ticker)

result_df = pd.DataFrame(columns=['Stock', 'Monthly Return'])

data = [{'Stock': ticker, 'Monthly Return': monthly_return} for ticker, monthly_return in stock_dataframes.items()]

result_df = pd.DataFrame(data)

def highest_average_combination(stock_returns):
    max_average = float('-inf')
    best_combination = []

    for r in range(1, len(stock_returns) + 1):
        for combo in combinations(stock_returns, r):
            avg = sum(combo) / len(combo)
            if avg > max_average:
                max_average = avg
                best_combination = combo

    return best_combination, max_average

stock_returns = result_df['Monthly Return'].tolist()
tickers = result_df['Stock'].tolist()

best_combination, max_avg = highest_average_combination(stock_returns)

best_tickers = [tickers[stock_returns.index(return_val)] for return_val in best_combination]

best_combination, max_avg = highest_average_combination(stock_returns)

best_tickers = [tickers[stock_returns.index(return_val)] for return_val in best_combination]

best_combination_df = pd.DataFrame({
    'Stock': best_tickers,
    'Monthly Return': best_combination
})

file_path = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\best_combination.csv'
best_combination_df.to_csv(file_path, index=False)

print(f"Best combination saved to '{file_path}'")
print(f"Highest average: {max_avg}")