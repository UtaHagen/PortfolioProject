import yfinance as yf
import pandas as pd
from datetime import datetime

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

stock_list_path = r'Data\sp500company.csv'
stock_list = pd.read_csv(stock_list_path)

stock_dataframes = {}

excluded_tickers = ['BRK.B', 'BF.B']

for ticker in stock_list['Symbol']:
    if ticker in excluded_tickers:
        continue
    try:
        stock_dataframes[ticker] = calculate_most_recent_monthly_return(ticker)
    except Exception:
        continue
recent_monthly_returns_df = pd.DataFrame.from_dict(stock_dataframes, orient='index', columns=['Recent Monthly Return'])

top_10_tickers = recent_monthly_returns_df.nlargest(10, 'Recent Monthly Return')


average_return = top_10_tickers['Recent Monthly Return'].mean()

top_10_info = {}
for ticker in top_10_tickers.index:
    stock = yf.Ticker(ticker)
    info = stock.info
    top_10_info[ticker] = info

top_10_info_df = pd.DataFrame.from_dict(top_10_info, orient='index')

top_10_info_df.to_csv(r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\top 10 info.csv', index=False)