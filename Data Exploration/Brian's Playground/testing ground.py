import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

file_path = r'Data\portfolio_pick.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

tickers = df['Stock'].tolist()

stock_data = {}

# Step 4: Download financial data for each ticker
for ticker in tickers:
    try:
        data = yf.download(ticker, period="1y")  # Download 1 year of data, change as needed
        stock_data[ticker] = data
    except Exception as e:
        print(f"Could not download data for {ticker}: {e}")


print(stock_data)