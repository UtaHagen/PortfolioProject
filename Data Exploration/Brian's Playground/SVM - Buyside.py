import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

def load_stock_data_from_csv(file_path):
    """
    Load stock data from a CSV file with columns: Date,Price,Close,High,Low,Open,Volume
    """
    try:
        # Read the CSV file
        stock_data = pd.read_csv(file_path)
        
        # Convert Date column to datetime
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Set Date as index
        stock_data = stock_data.set_index('Date')
        
        # Sort by date (ascending)
        stock_data = stock_data.sort_index()
        
        # Extract ticker symbol from filename
        ticker = os.path.basename(file_path).split('.')[0]
        
        print(f"Successfully loaded data for {ticker}")
        return stock_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_features(stock_data, n_forward=10):
    """
    Create features for the SVM model:
    - Technical indicators
    - Price patterns
    - Target variable: 1 if the price increases by X% in n_forward days, 0 otherwise
    """
    # Copy the data
    df = stock_data.copy()
    
    # Ensure all required columns exist
    required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: {col} column missing from data. Some features will not be calculated.")
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Create features
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Price relative to moving averages
    df['Price_to_MA5'] = df['Close'] / df['MA5']
    df['Price_to_MA10'] = df['Close'] / df['MA10']
    df['Price_to_MA20'] = df['Close'] / df['MA20']
    df['Price_to_MA50'] = df['Close'] / df['MA50']
    df['Price_to_MA200'] = df['Close'] / df['MA200']
    
    # Moving average crossovers
    df['MA5_cross_MA10'] = (df['MA5'] > df['MA10']).astype(int)
    df['MA10_cross_MA20'] = (df['MA10'] > df['MA20']).astype(int)
    df['MA20_cross_MA50'] = (df['MA20'] > df['MA50']).astype(int)
    df['MA50_cross_MA200'] = (df['MA50'] > df['MA200']).astype(int)
    
    # Volatility measures
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # True Range and Average True Range (ATR)
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        df['PrevClose'] = df['Close'].shift(1)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['PrevClose']),
                abs(df['Low'] - df['PrevClose'])
            )
        )
        df['ATR14'] = df['TR'].rolling(window=14).mean()
        df['ATR_Ratio'] = df['ATR14'] / df['Close']
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Hist_Positive'] = (df['MACD_Hist'] > 0).astype(int)
    df['MACD_Cross_Signal'] = ((df['MACD'] > df['MACD_Signal']) & 
                               (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume indicators
    if 'Volume' in df.columns:
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        df['Relative_Volume'] = df['Volume'] / df['Volume_MA10']
        df['Volume_Price_Trend'] = (df['Volume'] * (df['Close'] - df['Close'].shift(1))).cumsum()
        df['OBV'] = (df['Volume'] * np.where(df['Close'] > df['Close'].shift(1), 1, 
                                           np.where(df['Close'] < df['Close'].shift(1), -1, 0))).cumsum()
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
        df['OBV_Signal'] = (df['OBV'] > df['OBV_MA20']).astype(int)
    
    # Candle pattern features
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        df['Body_Size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'])
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'])
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        df['Gap_Up'] = (df['Low'] > df['High'].shift(1)).astype(int)
        df['Gap_Down'] = (df['High'] < df['Low'].shift(1)).astype(int)
    
    # Advanced trend indicators
    df['Returns_5d'] = df['Close'].pct_change(periods=5)
    df['Returns_10d'] = df['Close'].pct_change(periods=10)
    df['Returns_20d'] = df['Close'].pct_change(periods=20)
    
    # Target: Will price increase by 2% in the next n_forward days?
    future_price = df['Close'].shift(-n_forward)
    df['Target'] = ((future_price - df['Close']) / df['Close'] > 0.02).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def train_svm_model(features_df, feature_columns=None, target_column='Target'):
    """
    Train an SVM model on the features
    """
    if feature_columns is None:
        # Basic feature set
        basic_features = [
            'Price_to_MA5', 'Price_to_MA10', 'Price_to_MA20', 'Price_to_MA50', 'Price_to_MA200',
            'MA5_cross_MA10', 'MA10_cross_MA20', 'MA20_cross_MA50', 'MA50_cross_MA200',
            'Volatility', 'RSI', 'MACD_Hist_Positive', 'MACD_Cross_Signal',
            'BB_Position', 'BB_Width'
        ]
        
        # Add volume features if available
        volume_features = [
            'Volume_Change', 'Relative_Volume', 'OBV_Signal'
        ]
        
        # Add candle pattern features if available
        candle_features = [
            'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Is_Bullish', 'Gap_Up', 'Gap_Down'
        ]
        
        # Check which features exist in the dataframe
        feature_columns = [col for col in basic_features if col in features_df.columns]
        feature_columns += [col for col in volume_features if col in features_df.columns]
        feature_columns += [col for col in candle_features if col in features_df.columns]
    
    # Check if we have enough features
    if len(feature_columns) < 5:
        print("Warning: Not enough features available. Model may not perform well.")
    
    # Print features being used
    print(f"Training model with {len(feature_columns)} features:")
    for i, feature in enumerate(feature_columns):
        print(f"  {i+1}. {feature}")
    
    # Prepare data
    X = features_df[feature_columns]
    y = features_df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Check for class imbalance
    class_counts = y_train.value_counts()
    print(f"\nClass distribution in training data:")
    print(f"  No Buy: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y_train):.1%})")
    print(f"  Buy: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y_train):.1%})")
    
    # If severe class imbalance, adjust class weights
    class_weight = None
    if class_counts.get(1, 0) / len(y_train) < 0.2:
        print("Class imbalance detected. Using balanced class weights.")
        class_weight = 'balanced'
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search for best parameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear', 'poly'],
        'class_weight': [class_weight, None]
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True), 
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # Using F1 score for imbalanced data
        verbose=1
    )
    
    print("\nPerforming grid search to find optimal parameters...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test data
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Print evaluation metrics
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Buy', 'Buy'])
    plt.yticks(tick_marks, ['No Buy', 'Buy'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center", 
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Save the figure instead of showing it
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')  # Save the ROC curve plot
    
    # Feature importance analysis
    if best_model.kernel == 'linear':
        # For linear SVM, we can get feature importance from weights
        importance = np.abs(best_model.coef_[0])
        feature_importance = pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('feature_importance.png')  # Save the feature importance plot
        
    return best_model, scaler, feature_columns

def predict_buy_signal(model, scaler, new_data, feature_columns):
    """
    Predict buy signals for new data
    """
    # Prepare the data
    X_new = new_data[feature_columns]
    X_new_scaled = scaler.transform(X_new)
    
    # Get probabilities
    probabilities = model.predict_proba(X_new_scaled)
    
    # Get binary predictions
    predictions = model.predict(X_new_scaled)
    
    # Add predictions to the dataframe
    new_data['Buy_Probability'] = probabilities[:, 1]
    new_data['Buy_Signal'] = predictions
    
    return new_data

def plot_signals(stock_data_with_signals, start_idx=None, end_idx=None, ticker='Stock', save_path=None):
    """
    Plot the stock price with buy signals
    """
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(stock_data_with_signals)
        
    plot_data = stock_data_with_signals.iloc[start_idx:end_idx]
    
    plt.figure(figsize=(14, 10))
    
    # Plot stock price
    plt.subplot(3, 1, 1)
    plt.plot(plot_data.index, plot_data['Close'], label='Close Price')
    
    # Plot MA lines
    if 'MA20' in plot_data.columns:
        plt.plot(plot_data.index, plot_data['MA20'], 'g--', alpha=0.6, label='MA20')
    if 'MA50' in plot_data.columns:
        plt.plot(plot_data.index, plot_data['MA50'], 'r--', alpha=0.6, label='MA50')
    
    # Plot buy signals
    buy_signals = plot_data[plot_data['Buy_Signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], 
                color='green', marker='^', s=100, label='Buy Signal')
    
    plt.title(f'{ticker} - Price with Buy Signals')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot technical indicators
    plt.subplot(3, 1, 2)
    if 'RSI' in plot_data.columns:
        plt.plot(plot_data.index, plot_data['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    if 'MACD' in plot_data.columns and 'MACD_Signal' in plot_data.columns:
        plt.plot(plot_data.index, plot_data['MACD'], label='MACD')
        plt.plot(plot_data.index, plot_data['MACD_Signal'], label='Signal')
    
    #plt.title('Technical Indicators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot buy probabilities
    plt.subplot(3, 1, 3)
    plt.plot(plot_data.index, plot_data['Buy_Probability'], color='blue', label='Buy Probability')
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3, label='Threshold')
    plt.title('Buy Probability')
    plt.ylabel('Probability')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        filename = os.path.join(save_path, f"{ticker}_signals_plot.png")
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
    
    plt.close()  # Close the plot to free memory

def analyze_trading_performance(stock_data_with_signals, initial_capital=10000, commission=0.001, save_path=None, ticker='Stock'):
    """
    Analyze the performance of the trading strategy and save performance graphs.
    """
    # Create a copy of the data
    df = stock_data_with_signals.copy()
    
    # Initialize columns
    df['Position'] = 0  # 0: no position, 1: long position
    df['Entry_Price'] = 0.0
    df['Exit_Price'] = 0.0
    df['Trade_Return'] = 0.0
    
    # Simulate trading based on buy signals
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(1, len(df)):
        # If no position and buy signal, enter position
        if position == 0 and df.iloc[i-1]['Buy_Signal'] == 1:
            position = 1
            entry_price = df.iloc[i]['Close']
            df.iloc[i, df.columns.get_loc('Position')] = 1
            df.iloc[i, df.columns.get_loc('Entry_Price')] = entry_price
        
        # If in position and no more buy signal or end of data, exit position
        elif position == 1 and (df.iloc[i-1]['Buy_Signal'] == 0 or i == len(df) - 1):
            position = 0
            exit_price = df.iloc[i]['Close']
            trade_return = (exit_price / entry_price) - 1 - (2 * commission)  # Account for buy and sell commission
            
            df.iloc[i, df.columns.get_loc('Position')] = 0
            df.iloc[i, df.columns.get_loc('Exit_Price')] = exit_price
            df.iloc[i, df.columns.get_loc('Trade_Return')] = trade_return
            
            trades.append({
                'Entry_Date': df.iloc[i-1].name,
                'Exit_Date': df.iloc[i].name,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Return': trade_return
            })
    
    # Calculate performance metrics
    if trades:
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate equity curve
        equity = [initial_capital]
        for ret in trades_df['Return']:
            equity.append(equity[-1] * (1 + ret))
        
        # Calculate drawdown
        max_equity = pd.Series(equity).cummax()
        drawdown = (pd.Series(equity) / max_equity - 1) * 100
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Capital ($)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            equity_curve_path = os.path.join(save_path, f"{ticker}_equity_curve.png")
            plt.savefig(equity_curve_path)
            print(f"Saved equity curve to {equity_curve_path}")
        
        plt.close()
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        plt.plot(drawdown, color='red')
        plt.title('Drawdown Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            drawdown_curve_path = os.path.join(save_path, f"{ticker}_drawdown_curve.png")
            plt.savefig(drawdown_curve_path)
            print(f"Saved drawdown curve to {drawdown_curve_path}")
        
        plt.close()
        
        return trades_df, equity[-1] / equity[0] - 1
    else:
        print("\nNo trades executed during the period.")
        return None, 0

def run_stock_svm_analysis(csv_folder_path, stock_filename, prediction_days=10):
    """
    Run the complete SVM analysis pipeline for a specific stock
    """
    # Extract ticker from filename
    ticker = os.path.splitext(stock_filename)[0]
    file_path = os.path.join(csv_folder_path, stock_filename)
    
    # Load data
    print(f"Loading data for {ticker}...")
    stock_data = load_stock_data_from_csv(file_path)
    
    if stock_data is None:
        print(f"Failed to load data for {ticker}. Skipping analysis.")
        return None
    
    # Create features
    print("Creating features...")
    features_df = create_features(stock_data, n_forward=prediction_days)
    
    # Check if we have enough data after feature creation
    if len(features_df) < 100:
        print(f"Warning: Limited data available for {ticker} after feature creation. Model may not be reliable.")
    
    # Train model
    print("Training SVM model...")
    model, scaler, feature_columns = train_svm_model(features_df)
    
    # Make predictions on the whole dataset
    print("Generating buy signals...")
    stock_data_with_signals = predict_buy_signal(model, scaler, features_df, feature_columns)
    
    # Plot the results
    print("Plotting results...")
    # Plot last 100 days
    if len(stock_data_with_signals) > 100:
        plot_signals(stock_data_with_signals, -100, ticker=ticker)
    else:
        plot_signals(stock_data_with_signals, ticker=ticker)
    
    # Analyze the most recent data points
    recent_signals = stock_data_with_signals.tail(5)
    print("\nRecent signals:")
    print(recent_signals[['Close', 'Buy_Probability', 'Buy_Signal']])
    
    # Analyze trading performance
    print("\nAnalyzing trading performance...")
    trades, total_return = analyze_trading_performance(stock_data_with_signals)
    
    # Calculate accuracy metrics for buy signals
    future_returns = stock_data_with_signals['Target']
    predicted_buys = stock_data_with_signals['Buy_Signal']
    
    correct_buys = (predicted_buys == 1) & (future_returns == 1)
    incorrect_buys = (predicted_buys == 1) & (future_returns == 0)
    
    if sum(predicted_buys) > 0:
        accuracy = sum(correct_buys) / sum(predicted_buys)
        print(f"\nBuy signal accuracy: {accuracy:.2%}")
    else:
        print("\nNo buy signals generated in this period.")
    
    return model, scaler, feature_columns, stock_data_with_signals

def batch_analyze_stocks(csv_folder_path, prediction_days=10):
    """
    Analyze all stock CSV files in the folder
    """
    # Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_folder_path}")
        return
    
    print(f"Found {len(csv_files)} stock files to analyze")
    
    # Results storage
    results = {}
    
    # Path to save results
    results_folder_path = os.path.join(csv_folder_path, "..", "SVM Results")
    os.makedirs(results_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Process each file
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing {csv_file}...")
        print(f"{'='*50}")
        
        model, scaler, features, signals = run_stock_svm_analysis(
            csv_folder_path=csv_folder_path,
            stock_filename=csv_file,
            prediction_days=prediction_days
        )
        
        # Store results
        ticker = os.path.splitext(csv_file)[0]
        if model is not None and signals is not None:
            recent_signal = signals.iloc[-1]['Buy_Signal']
            recent_prob = signals.iloc[-1]['Buy_Probability']
            results[ticker] = {
                'buy_signal': recent_signal,
                'probability': recent_prob,
                'model': model,
                'scaler': scaler,
                'features': features
            }

            # Save model, scaler, and signals to files
            model_file = os.path.join(results_folder_path, f"{ticker}_model.pkl")
            scaler_file = os.path.join(results_folder_path, f"{ticker}_scaler.pkl")
            signals_file = os.path.join(results_folder_path, f"{ticker}_signals.csv")
            
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            signals.to_csv(signals_file, index=True)
            print(f"Saved results for {ticker} to {results_folder_path}")
    
    # Print summary
    print("\n===== Analysis Summary =====")
    for ticker, result in results.items():
        signal_text = "BUY" if result['buy_signal'] == 1 else "No action"
        print(f"{ticker}: {signal_text} (Probability: {result['probability']:.2f})")
    
    return results

if __name__ == "__main__":
    # Path to your CSV files
    csv_folder_path = r"Data Exploration\Brian's Playground\Testing Data"
    
    # Run batch analysis on all stocks
    results = batch_analyze_stocks(csv_folder_path, prediction_days=10)