#This is a monte carlo stim - functional but incomplete, the graph is trunc to a specific period and I haven't figured out how to fix it yet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

# Download historical data as dataframe
ticker = "^GSPC"
data = pdr.get_data_yahoo(ticker, start="2020-01-01", end="2022-12-31")

# Calculate the log returns
data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

# Build the Monte Carlo simulation function
def monte_carlo_simulation(data, num_simulations, num_days):
    simulations = np.zeros((num_simulations, num_days))
    daily_returns = data['log_return'].dropna()
    
    # Compute the cumulative return
    for simulation in range(num_simulations):
        price_series = []
        price = data['Adj Close'].iloc[-1]  # Use iloc for position-based indexing
        price_series.append(price)
        
        # Generate the price series
        for i in range(num_days - 1):  # Subtract 1 to match the shape of simulations
            price = price_series[-1] * np.exp(np.random.normal(daily_returns.mean(), daily_returns.std()))
            price_series.append(price)
        
        simulations[simulation, :] = price_series
    
    return simulations

# Run the Monte Carlo simulator
num_simulations = 2500
num_days = 252
simulations = monte_carlo_simulation(data, num_simulations, num_days)

# Calculate the returns of the simulations
returns = simulations / simulations[:, 0, None] - 1

# Calculate the minimum and maximum returns
min_returns = np.min(returns, axis=0)
max_returns = np.max(returns, axis=0)

# Plot the results
plt.figure(figsize=(10,5))

# Plot the historical equity curve
plt.plot((data['Adj Close'] / data['Adj Close'].iloc[0] - 1).values, label='Historical Equity Curve')

# Plot the full range of return streams
plt.fill_between(range(num_days), min_returns, max_returns, color='b', alpha=0.3)

plt.xlabel("Trading Days")
plt.ylabel("Return")
plt.title(f"Monte Carlo Simulation for {ticker}")
plt.legend()
plt.show()