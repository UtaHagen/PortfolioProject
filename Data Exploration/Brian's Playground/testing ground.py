import os
import pandas as pd
import yfinance as yf

file_path = r'Data\portfolio_pick.csv'
df_tickers = pd.read_csv(file_path)
tickers = df_tickers['Stock'].tolist()
base_folder = r"Data Exploration\Brian's Playground"
subfolder = os.path.join(base_folder, "Testing Data")

# Create subfolder if it doesn't exist
os.makedirs(subfolder, exist_ok=True)

data = yf.download(tickers, start='2015-01-01', end='2021-09-12', group_by='ticker')

for ticker in tickers:
    if ticker in data:  # Ensure the ticker exists in the dataset
        stock_data = data[ticker].copy()  # Extract data for the specific ticker
        
        # Check if "Adj Close" exists, otherwise use "Close"
        if 'Adj Close' in stock_data.columns:
            stock_data.insert(0, 'Price', stock_data['Adj Close'])
        elif 'Close' in stock_data.columns:
            stock_data.insert(0, 'Price', stock_data['Close'])
        else:
            print(f"Skipping {ticker}: No 'Adj Close' or 'Close' column found.")
            continue  # Skip to the next ticker if neither column exists

        # Ensure correct column order
        required_columns = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
        available_columns = [col for col in required_columns if col in stock_data.columns]
        stock_data = stock_data[available_columns]  # Keep only available columns

        # Define the full CSV file path inside the "Testing Data" subfolder
        file_path = os.path.join(subfolder, f"{ticker}.csv")

        # Save to CSV
        stock_data.to_csv(file_path, index=True)