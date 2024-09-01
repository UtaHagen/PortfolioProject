from itertools import combinations
from fractions import Fraction
import yfinance as yf
import pandas as pd

stock_returns = [0.1,0.2,-0.2,-1,1,2]

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