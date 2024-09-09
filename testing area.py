import yfinance as yf
import pandas as pd
from datetime import datetime

monthly_returns_path = r'Data\portfolio_pick.csv'
monthly_returns = pd.read_csv(monthly_returns_path)

def calculate_most_recent_monthly_return(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2mo")
    hist['Monthly Return'] = hist['Close'].pct_change(periods=21)
    hist = hist[['Monthly Return']].dropna()

    today = datetime.today()
    if today.day == 1:
        most_recent_return = hist.iloc[-1]
    else:
        most_recent_return = hist.iloc[-2]

    return most_recent_return['Monthly Return']

stock_dataframes = {}

for ticker in monthly_returns['Stock']:
    stock_dataframes[ticker] = calculate_most_recent_monthly_return(ticker)

result_df = pd.DataFrame(columns=['Stock', 'Monthly Return'])

data = [{'Stock': ticker, 'Monthly Return': monthly_return} for ticker, monthly_return in stock_dataframes.items()]

result_df = pd.DataFrame(data)