import yfinance as yf
import pandas as pd
from datetime import datetime

# Load the portfolio data
monthly_returns_path = r'Data\portfolio_pick.csv'
monthly_returns = pd.read_csv(monthly_returns_path)

def calculate_most_recent_monthly_return(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2mo")
    hist['Monthly Return'] = hist['Close'].pct_change(periods=21)
    hist = hist[['Monthly Return']].dropna()

    today = datetime.today()
    most_recent_return = hist.iloc[-1] if today.day == 1 else hist.iloc[-2]

    return most_recent_return['Monthly Return']

# Calculate monthly returns for each stock
stock_dataframes = {ticker: calculate_most_recent_monthly_return(ticker) for ticker in monthly_returns['Stock']}

# Create a DataFrame with the results
result_df = pd.DataFrame(list(stock_dataframes.items()), columns=['Stock', 'Monthly Return'])

# Get the top 10 tickers with the highest returns
top_10_df = result_df.nlargest(10, 'Monthly Return')

# Output the result to a CSV file
top_10_df.to_csv('top10portfoliopick.csv', index=False)