import pandas as pd
import numpy as np

portfolio_pick = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\portfolio_pick.csv'
portfolio_pick_df = pd.read_csv(portfolio_pick)
# Assuming your DataFrame is named 'df' and has columns 'Date', 'Ticker', and 'Close'
# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter data for the last 5 years
end_date = df['Date'].max()
start_date = end_date - pd.DateOffset(years=5)
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Set 'Date' as index
df.set_index('Date', inplace=True)

# Calculate monthly return
monthly_returns = df.groupby('Ticker')['Close'].resample('M').ffill().pct_change().dropna()

# Reset index to have 'Date' and 'Ticker' as columns again
monthly_returns = monthly_returns.reset_index()

# Rename the 'Close' column to 'Monthly Return'
monthly_returns.rename(columns={'Close': 'Monthly Return'}, inplace=True)

# Create a new DataFrame to store the monthly returns
monthly_returns_df = pd.DataFrame(monthly_returns)

print(monthly_returns_df.head())