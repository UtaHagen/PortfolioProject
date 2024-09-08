from itertools import combinations
from fractions import Fraction
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

monthly_returns_path = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\monthly_returns.csv'
monthly_returns = pd.read_csv(monthly_returns_path)

stock_returns = monthly_returns['Monthly Return'].dropna().tolist()
tickers = monthly_returns['Ticker'].tolist()

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

best_combination, max_avg = highest_average_combination(stock_returns)

# Find the tickers corresponding to the best combination
best_tickers = [tickers[stock_returns.index(return_val)] for return_val in best_combination]

# Create a DataFrame for the best combination
best_combination_df = pd.DataFrame({
    'Ticker': best_tickers,
    'Monthly Return': best_combination
})

# Save the DataFrame to a CSV file
file_path = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\best_combination.csv'
best_combination_df.to_csv(file_path, index=False)

print(f"Best combination saved to '{file_path}'")
print(f"Highest average: {max_avg}")