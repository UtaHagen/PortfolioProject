import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


file_path = r'Data\portfolio_pick.csv'
file_data = pd.read_csv(file_path)

tickers = file_data['Stock'].tolist()

stock_data = []

# Download stock data for each ticker
for ticker in tickers:
    try:
        data = yf.download(ticker, period="5y")
        stock_data[ticker] = data
    except Exception as e:
        print(f"Could not download data for {ticker}: {e}")

# Initialize variables
n_forward = 10

for ticker, stock_df in stock_data.items():  # Iterate over each stock's data frame
    try:
        print(f"\n📊 Running SVM analysis for {ticker}...")

        # Check if the 'Close' column exists
        if 'Close' not in stock_df.columns:
            raise ValueError(f"Close column not found in data for {ticker}")

        # Create the feature dataframe
        df = stock_df[['Close']].copy()  # Only use 'Close' column initially

        # Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()

        # Price relative to MAs
        for window in [5, 10, 20, 50]:
            df[f'Price_to_MA{window}'] = df['Close'] / df[f'MA{window}']

        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()

        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))

        # Target: Will price increase by 2% in next `n_forward` days?
        df['Target'] = (df['Close'].shift(-n_forward) > df['Close'] * 1.02).astype(int)

        features_df = df.dropna()

        feature_columns = [col for col in features_df.columns if col not in ['Target']]

        X = features_df[feature_columns]
        y = features_df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Scaling features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Grid Search for SVM hyperparameters
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 0.01], 'kernel': ['rbf']}

        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        # Print results
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        # Predict Buy Signals for the whole dataset
        X_scaled = scaler.transform(X)
        features_df['Buy_Probability'] = best_model.predict_proba(X_scaled)[:, 1]
        features_df['Buy_Signal'] = best_model.predict(X_scaled)

        print("\nRecent Buy Signals:")
        print(features_df[['Close', 'Buy_Probability', 'Buy_Signal']].tail(5))

    except Exception as e:
        print(f"❌ ERROR: {e}")