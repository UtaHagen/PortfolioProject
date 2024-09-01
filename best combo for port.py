from itertools import combinations
from fractions import Fraction
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging



portfolio_pick = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\portfolio_pick.csv'
portfolio_pick_df = pd.read_csv(portfolio_pick)

stock_returns = 

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

print(f"Best combination: {best_combination}")
print(f"Highest average: {max_avg}")