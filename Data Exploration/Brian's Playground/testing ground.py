import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


file_path = r'Data\portfolio_pick.csv'
df_tickers = pd.read_csv(file_path)  # Replace with your actual file path
tickers = df_tickers['Stock'].tolist()  # Replace 'Stock' with the actual column name in your CSV

# Initialize an empty list to store data
stock_data = []

# Download historical data for each ticker
for ticker in tickers:
    data = yf.download(ticker, start="2020-01-01", end="2025-01-01")  # Specify date range as needed
    data['Stock'] = ticker  # Add the ticker to each row of data
    stock_data.append(data)

# Concatenate all the stock data into a single dataframe
final_df = pd.concat(stock_data)

# Reset the index to make the 'Date' column a regular column
final_df.reset_index(inplace=True)

# Preview the final dataframe
print(final_df.head())