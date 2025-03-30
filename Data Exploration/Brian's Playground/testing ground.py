import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def load_stock_data(file_path, start_date="2020-01-01", end_date=None):
    """
    Load stock data from yfinance for tickers listed in the given CSV file.
    Each ticker has its own DataFrame stored in a dictionary.
    """
    try:
        df = pd.read_csv(file_path)

        if 'Stock' not in df.columns:
            raise ValueError("❌ ERROR: 'Stock' column not found in the CSV file.")

        tickers = df['Stock'].dropna().unique().tolist()  # Remove duplicates & NaN values
        if not tickers:
            raise ValueError("❌ ERROR: No valid tickers found in the CSV file.")

        print(f"📥 Fetching data for: {', '.join(tickers)}")

        # Download stock data
        stock_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, group_by='ticker')

        # Create a dictionary to store each stock's data separately
        stock_dict = {}

        for ticker in tickers:
            try:
                if isinstance(stock_data, dict):
                    stock_df = stock_data[ticker]
                else:
                    stock_df = stock_data.xs(ticker, axis=1, level=0)  # Extract per-ticker data

                stock_df = stock_df[['Adj Close']].rename(columns={'Adj Close': 'Close'})  # Keep only 'Close'
                stock_dict[ticker] = stock_df  # Store in dictionary
            except KeyError:
                print(f"⚠ Warning: No data found for {ticker}, skipping.")

        print("✅ Successfully loaded stock data.")
        return stock_dict

    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    filepath = r"Data/portfolio_pick.csv"
    load_stock_data(filepath)