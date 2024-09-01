import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(get_beta, dirty_df['Stock']))

#beta_dict = dict(results)

dirty_df['Beta'] = dirty_df['Stock'].apply(lambda x: yf.Ticker(x).info.get('beta', None))

research_df = dirty_df.dropna(subset=['Beta'])

portfolio_data = []

for index, row in research_df.iterrows():
    if row['Coefficient'] < 0 and row['Beta'] < 0:
        portfolio_data.append(row)
    elif row['Coefficient'] > 0 and row['Beta'] > 0:
        portfolio_data.append(row)

portfolio_pick = pd.concat([pd.DataFrame(portfolio_data)], ignore_index=True)

portfolio_pick.to_csv(r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\portfolio_pick.csv', index=False)