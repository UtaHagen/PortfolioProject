import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def load_stock_data(file_path, start_date, end_date=None):
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


def create_features(stock_data, n_forward=10):
    """Creates technical indicators and target variable."""
    df = stock_data['Close'].copy()

    # Convert to DataFrame if it's a Series (single ticker case)
    if isinstance(df, pd.Series):
        df = df.to_frame()

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

    return df.dropna()



def train_svm_model(features_df):
    """Trains an SVM model with grid search."""
    feature_columns = [col for col in features_df.columns if col not in ['Target']]
    
    X = features_df[feature_columns]
    y = features_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 0.01], 'kernel': ['rbf']}
    
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    return best_model, scaler, feature_columns


def predict_buy_signals(model, scaler, new_data, feature_columns):
    """Predicts buy signals for new data."""
    X_new = new_data[feature_columns]
    X_scaled = scaler.transform(X_new)
    
    new_data['Buy_Probability'] = model.predict_proba(X_scaled)[:, 1]
    new_data['Buy_Signal'] = model.predict(X_scaled)
    
    return new_data


def run_stock_svm_analysis(filepath, start_date="2020-01-01", prediction_days=10):
    """Runs the full SVM analysis pipeline."""
    try:
        stock_dict = load_stock_data(filepath, start_date)  # Load stock data dictionary

        for ticker, stock_data in stock_dict.items():
            print(f"\n📊 Running SVM analysis for {ticker}...")

            # Feature Engineering
            features_df = create_features(stock_data, n_forward=prediction_days)

            # Train Model
            model, scaler, feature_columns = train_svm_model(features_df)

            # Predict Buy Signals
            signals = predict_buy_signals(model, scaler, features_df, feature_columns)

            print("\nRecent Buy Signals:")
            print(signals[['Close', 'Buy_Probability', 'Buy_Signal']].tail(5))
        
        return stock_dict  # Return stock data dictionary for further analysis

    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    filepath = r"Data/portfolio_pick.csv"
    run_stock_svm_analysis(filepath)