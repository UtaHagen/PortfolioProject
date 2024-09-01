import yfinance as yf
import pandas as pd

results_path = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\results.csv'
sectors_path = r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\sectorsxd - Sheet1.csv'
results_df = pd.read_csv(results_path)
sectors_df = pd.read_csv(sectors_path)

dirty_df = pd.merge(results_df, sectors_df, on='Sector', how='inner')

def get_beta(ticker):
    try:
        stock = yf.Ticker(ticker)
        beta = stock.info.get('beta', None)
        return beta
    except Exception as e:
        if '404' in str(e):
            print(f"404 error for {ticker}: {e}")
            return None
        else:
            print(f"Error fetching data for {ticker}: {e}")
            return None

dirty_df['Beta'] = dirty_df['Stock'].apply(lambda x: yf.Ticker(x).info.get('beta', None))

#cleaned_df = merged_df.dropna(subset=['Beta'])

research_df = dirty_df.dropna()

print(research_df)