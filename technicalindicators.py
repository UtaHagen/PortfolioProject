#!/usr/bin/env python
# coding: utf-8

# In[4]:

#This script calculates technical indicators for each S&P 500 ticker.


#could not find current web reference for ROC,CMO,PPO, Shiller's PE to check calculation output 
#maybe double check those, first time calculating them
#the other outputs were verified with below websites
#https://www.tradingview.com/ideas/oscillator/ and https://www.barchart.com/stocks/quotes/AAPL/technical-analysis#google_vignette

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import numpy as np

url = 'https://stockanalysis.com/list/sp-500-stocks/'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

tickers = []
table = soup.find('table')  
for row in table.find_all('tr')[1:]:  
    ticker = row.find('a').text  
    tickers.append(ticker.strip())

print(f"Extracted {len(tickers)} tickers.")

def SMA(data, ndays): 
    return data['Close'].rolling(ndays).mean()

def EWMA(data, ndays): 
    return data['Close'].ewm(span=ndays, min_periods=ndays - 1).mean()

def rsi(data, periods=14):
    close_delta = data.diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def stochastic_oscillator(data, k_window=14, d_window=3):
    data['Low_Min'] = data['Low'].rolling(window=k_window).min()
    data['High_Max'] = data['High'].rolling(window=k_window).max()
    data['%K'] = 100 * (data['Close'] - data['Low_Min']) / (data['High_Max'] - data['Low_Min'])
    data['%K'] = data['%K'].rolling(window=d_window).mean()  
    return data['%K']

def williams_r(data, lookback=14):
    highh = data['High'].rolling(lookback).max() 
    lowl = data['Low'].rolling(lookback).min()
    wr = -100 * ((highh - data['Close']) / (highh - lowl))
    return wr

def get_macd(data, slow=26, fast=12, smooth=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=smooth, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def get_cmo(data, lookback=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    sum_gain = gain.rolling(window=lookback).sum()
    sum_loss = loss.rolling(window=lookback).sum()
    cmo = 100 * ((sum_gain - sum_loss) / (sum_gain + sum_loss))
    return cmo

def ppo(data, sm=12, lm=26):
    sema = data.ewm(span=sm, min_periods=sm, adjust=False).mean()
    lema = data.ewm(span=lm, min_periods=lm, adjust=False).mean()
    ppo = (sema - lema) / lema * 100
    signal_line = ppo.ewm(span=9, min_periods=9, adjust=False).mean()
    ppo_hist = ppo - signal_line
    return ppo, signal_line, ppo_hist

def get_typical_price(high, low, close):
    return (high + low + close) / 3.0

def mad(x):
    return np.mean(np.abs(x - np.mean(x)))

def run_cci(data, window=20, constant=0.015):
    typical_price = get_typical_price(data['High'], data['Low'], data['Close'])
    sma_tp = typical_price.rolling(window).mean()
    cci = (typical_price - sma_tp) / (constant * typical_price.rolling(window).apply(mad, True))
    return cci

def calculate_roc(data, period=14):
    roc = ((data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)) * 100
    return roc

def fetch_pe_ratio_yahoo(ticker):
    stock = yf.Ticker(ticker)
    pe_ratio = stock.info.get('trailingPE', None)
    return pe_ratio

def get_cape_ratio(data, pe_ratios, years=10):
    data = data.copy()
    data['Earnings'] = data['Close'] / pe_ratios
    rolling_earnings = data['Earnings'].rolling(window=years * 252).mean()  
    cape_ratio = data['Close'] / rolling_earnings
    return cape_ratio

start_date = "2010-01-01"  
end_date = datetime.now().strftime("%Y-%m-%d")
sma_period = 50
ewma_period = 200
rsi_period = 14
stochastic_k_window = 14
stochastic_d_window = 3
williams_r_lookback = 14
macd_slow = 26
macd_fast = 12
macd_smooth = 9
cmo_lookback = 14
ppo_sm = 12
ppo_lm = 26
cci_window = 20
roc_period = 14

results = pd.DataFrame(columns=[
    'Ticker', 'SMA', 'EWMA', 'RSI', 'Stochastic_%K', 
    'Williams_%R', 'MACD', 'MACD_Signal', 'MACD_Hist', 'CMO', 
    'PPO', 'PPO_Signal', 'PPO_Hist', 'CCI', 'ROC', 'CAPE'
])

for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            continue
        
        pe_ratio = fetch_pe_ratio_yahoo(ticker)
        if pe_ratio is None:
            continue
        
        pe_ratios = pd.Series(pe_ratio, index=data.index)
        
        # Calculate indicators
        data['SMA'] = SMA(data, sma_period)
        data['EWMA'] = EWMA(data, ewma_period)
        data['RSI'] = rsi(data['Close'], periods=rsi_period)
        data['Stochastic_%K'] = stochastic_oscillator(data, k_window=stochastic_k_window, d_window=stochastic_d_window)
        data['Williams_%R'] = williams_r(data, lookback=williams_r_lookback)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = get_macd(data['Close'], slow=macd_slow, fast=macd_fast, smooth=macd_smooth)
        data['CMO'] = get_cmo(data['Close'], lookback=cmo_lookback)
        data['PPO'], data['PPO_Signal'], data['PPO_Hist'] = ppo(data['Close'], sm=ppo_sm, lm=ppo_lm)
        data['CCI'] = run_cci(data, window=cci_window)
        data['ROC'] = calculate_roc(data, period=roc_period)
        data['CAPE'] = get_cape_ratio(data, pe_ratios, years=10)
        
        # Get the last row of indicators for the ticker
        last_row = data[['SMA', 'EWMA', 'RSI', 'Stochastic_%K', 'Williams_%R', 'MACD', 'MACD_Signal', 'MACD_Hist', 'CMO', 'PPO', 'PPO_Signal', 'PPO_Hist', 'CCI', 'ROC', 'CAPE']].iloc[-1]
        last_row['Ticker'] = ticker
        results = results.append(last_row, ignore_index=True)
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

print(results)


# In[ ]:




