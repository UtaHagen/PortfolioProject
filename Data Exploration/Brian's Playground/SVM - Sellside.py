import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from datetime import datetime

class StockSVMFromCSV:
    def __init__(self, filepath):
        self.filepath = filepath
        self.ticker = os.path.splitext(os.path.basename(filepath))[0]
        self.data = None
        self.model = None
        self.performance = None

    def fetch_data(self):
        df = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        required_columns = {'Close', 'High', 'Low', 'Open', 'Volume'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns in {self.filepath}")
        self.data = df
        return df

    def create_features(self, window_short=10, window_long=50, window_vol=20, future_days=5):
        df = self.data.copy()

        df['SMA_short'] = df['Close'].rolling(window=window_short).mean()
        df['SMA_long'] = df['Close'].rolling(window=window_long).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Volatility'] = df['Close'].pct_change().rolling(window=window_vol).std()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Momentum'] = df['Close'].pct_change(periods=10)
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=window_vol).mean()

        future_returns = df['Close'].shift(-future_days) / df['Close'] - 1
        df['Target'] = np.where(future_returns < -0.02, 1, 0)

        df.dropna(inplace=True)
        self.data = df
        return df

    def train_model(self):
        if 'Target' not in self.data.columns:
            self.create_features()

        features = ['SMA_short', 'SMA_long', 'RSI', 'Volatility', 'MACD', 
                    'MACD_signal', 'Momentum', 'Volume_Change', 'Volume_MA']

        X = self.data[features]
        y = self.data['Target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ])

        param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto'],
            'svm__kernel': ['rbf', 'linear']
        }

        print(f"Training model for {self.ticker}...")
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        y_pred = self.model.predict(X_test)

        print(f"Performance for {self.ticker}:")
        print(classification_report(y_test, y_pred))

        self.performance = {
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'best_params': grid.best_params_
        }

# Loop through CSVs in the folder
if __name__ == "__main__":
    folder_path = os.path.join("Data Exploration", "Brian's Playground", "Testing Data")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for file in csv_files:
        try:
            svm = StockSVMFromCSV(file)
            svm.fetch_data()
            svm.create_features()
            svm.train_model()
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            print("-" * 50)