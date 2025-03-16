#!/usr/bin/env python3
"""
Core-Satellite Capital Allocation Script with Sterling Ratio

This script implements a Core-Satellite Investment Approach to strategically allocate
capital among selected stocks. It ranks stocks based on Sterling ratio and price movements,
then allocates them to either Core or Satellite portfolios.

The script reads stock symbols from a CSV file and uses yfinance to fetch
historical data for analysis. The S&P 500 is used as the benchmark for all stocks.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
import yfinance as yf
import time

# Constants for portfolio allocation
CORE_ALLOCATION_RANGE = (0.70, 0.80)  # 70-80% for Core portfolio
SATELLITE_ALLOCATION_RANGE = (0.20, 0.30)  # 20-30% for Satellite portfolio
CORE_THRESHOLD = 0.5  # Threshold for Sterling ratio to be considered for Core portfolio

def read_stocks_from_csv(csv_path):
    """Read stock symbols and related data from CSV file."""
    stocks_data = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            stocks_data.append(row)
    
    print(f"Read {len(stocks_data)} stocks from CSV file")
    return stocks_data

def fetch_stock_data(symbol, period="3mo", interval="1d"):
    """Fetch historical stock data using yfinance."""
    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
        
        # Fetch data using yfinance
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        # Check if data was successfully retrieved
        if data.empty:
            print(f"Failed to fetch data for {symbol} (empty dataset)")
            return None
        
        # Ensure 'Adj Close' is used for calculations
        if 'Adj Close' not in data.columns:
            print(f"Warning: 'Adj Close' column not found for {symbol}")
            if 'Close' in data.columns:
                print(f"Using 'Close' as fallback for {symbol}")
                data['Adj Close'] = data['Close']
            else:
                print(f"Error: Neither 'Adj Close' nor 'Close' found for {symbol}")
                return None
        
        # Remove rows with NaN values
        data = data.dropna()
        
        print(f"Successfully fetched data for {symbol} with {len(data)} data points")
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_drawdowns(returns):
    """Calculate drawdowns from a series of returns."""
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdowns
    drawdowns = (cum_returns / running_max) - 1
    
    return drawdowns

def calculate_sterling_ratio(stock_data, benchmark_data, risk_free_rate=0.0):
    """Calculate Sterling ratio for a stock relative to the benchmark."""
    # Calculate daily returns
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
    
    # Align the return series
    common_dates = stock_returns.index.intersection(benchmark_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    # Calculate excess returns (stock returns - benchmark returns)
    excess_returns = stock_returns - benchmark_returns
    
    # Calculate annualized excess return
    annualized_excess_return = excess_returns.mean() * 252  # 252 trading days in a year
    
    # Calculate drawdowns
    drawdowns = calculate_drawdowns(excess_returns)
    
    # Calculate average maximum drawdown
    # For a 3-month period, we'll use the maximum drawdown
    max_drawdown = abs(drawdowns.min())
    
    # Apply the 10% adjustment as per Sterling ratio definition
    adjusted_drawdown = max(max_drawdown - 0.10, 0.01)  # Ensure we don't divide by zero
    
    # Calculate Sterling ratio
    sterling_ratio = annualized_excess_return / adjusted_drawdown
    
    # Calculate beta (using covariance and variance)
    beta = np.cov(stock_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
    
    return {
        'excess_return': annualized_excess_return,
        'max_drawdown': max_drawdown,
        'adjusted_drawdown': adjusted_drawdown,
        'sterling_ratio': sterling_ratio,
        'beta': beta,
        'is_high_sterling': sterling_ratio > CORE_THRESHOLD
    }

def calculate_price_movements(stock_data):
    """Calculate price movement metrics for a stock."""
    # Get current price (most recent adjusted close)
    current_price = stock_data['Adj Close'].iloc[-1]
    
    # Calculate 3-month low
    three_month_low = stock_data['Adj Close'].min()
    
    # Calculate distance from 3-month low (as percentage)
    distance_from_low = (current_price - three_month_low) / three_month_low * 100
    
    # Calculate price drop from 3-month high
    three_month_high = stock_data['Adj Close'].max()
    price_drop = (three_month_high - current_price) / three_month_high * 100
    
    return {
        'current_price': current_price,
        'three_month_low': three_month_low,
        'distance_from_low': distance_from_low,
        'three_month_high': three_month_high,
        'price_drop': price_drop
    }

def rank_stocks(stocks_analysis):
    """Rank stocks based on Sterling ratio and price movement metrics."""
    # Sort stocks by Sterling ratio (descending)
    sterling_sorted = sorted(stocks_analysis.items(), key=lambda x: x[1]['sterling']['sterling_ratio'], reverse=True)
    
    # Filter out stocks with high Sterling ratio
    high_sterling_stocks = [(symbol, data) for symbol, data in sterling_sorted 
                           if data['sterling']['is_high_sterling']]
    
    # Assign Sterling ratio ranks (1st place has 32 points, 2nd place has 31 points, etc.)
    total_stocks = len(stocks_analysis)
    for i, (symbol, _) in enumerate(sterling_sorted):
        stocks_analysis[symbol]['sterling_rank'] = total_stocks - i
        stocks_analysis[symbol]['sterling_rank_position'] = i + 1
    
    # Sort stocks by distance from 3-month low (ascending)
    low_price_sorted = sorted(stocks_analysis.items(), key=lambda x: x[1]['price_movements']['distance_from_low'])
    
    # Assign lowest price ranks
    for i, (symbol, _) in enumerate(low_price_sorted):
        stocks_analysis[symbol]['lowest_price_rank'] = total_stocks - i
        stocks_analysis[symbol]['lowest_price_rank_position'] = i + 1
    
    # Sort stocks by price drop (descending)
    price_drop_sorted = sorted(stocks_analysis.items(), key=lambda x: x[1]['price_movements']['price_drop'], reverse=True)
    
    # Assign price drop ranks
    for i, (symbol, _) in enumerate(price_drop_sorted):
        stocks_analysis[symbol]['price_drop_rank'] = total_stocks - i
        stocks_analysis[symbol]['price_drop_rank_position'] = i + 1
    
    return stocks_analysis

def allocate_portfolio(ranked_stocks):
    """Allocate stocks to Core and Satellite portfolios based on rankings."""
    # Calculate total points for each stock
    for symbol, data in ranked_stocks.items():
        # For stocks with high Sterling ratio, consider all metrics
        if data['sterling']['is_high_sterling']:
            data['total_points'] = (data['sterling_rank'] + data['lowest_price_rank'] + data['price_drop_rank']) / 3
            data['alpha_group'] = 'Alpha 1'  # Core candidates
        else:
            # For stocks with low Sterling ratio, focus more on price metrics
            data['total_points'] = (data['lowest_price_rank'] + data['price_drop_rank']) / 2
            data['alpha_group'] = 'Alpha 2'  # Satellite candidates
    
    # Sort stocks by total points (descending)
    sorted_stocks = sorted(ranked_stocks.items(), key=lambda x: x[1]['total_points'], reverse=True)
    
    # Separate Core and Satellite candidates
    core_candidates = [(symbol, data) for symbol, data in sorted_stocks if data['alpha_group'] == 'Alpha 1']
    satellite_candidates = [(symbol, data) for symbol, data in sorted_stocks if data['alpha_group'] == 'Alpha 2']
    
    # Determine allocation percentages
    total_allocation = 100.0
    core_allocation = total_allocation * CORE_ALLOCATION_RANGE[0]  # Start with minimum core allocation
    satellite_allocation = total_allocation - core_allocation
    
    # Allocate to Core portfolio (70-80% of total)
    total_core_points = sum(data['total_points'] for _, data in core_candidates)
    for symbol, data in core_candidates:
        if total_core_points > 0:
            core_percentage = (data['total_points'] / total_core_points) * core_allocation
            ranked_stocks[symbol]['core_allocation'] = round(core_percentage, 2)
            ranked_stocks[symbol]['satellite_allocation'] = 0.0
        else:
            ranked_stocks[symbol]['core_allocation'] = 0.0
            ranked_stocks[symbol]['satellite_allocation'] = 0.0
    
    # Allocate to Satellite portfolio (20-30% of total)
    total_satellite_points = sum(data['total_points'] for _, data in satellite_candidates)
    for symbol, data in satellite_candidates:
        if total_satellite_points > 0:
            satellite_percentage = (data['total_points'] / total_satellite_points) * satellite_allocation
            ranked_stocks[symbol]['core_allocation'] = 0.0
            ranked_stocks[symbol]['satellite_allocation'] = round(satellite_percentage, 2)
        else:
            ranked_stocks[symbol]['core_allocation'] = 0.0
            ranked_stocks[symbol]['satellite_allocation'] = 0.0
    
    return ranked_stocks

def create_output_dataframe(allocated_stocks):
    """Create a DataFrame with rankings and allocations."""
    output_data = []
    
    for symbol, data in allocated_stocks.items():
        sterling_ratio_value = round(data['sterling']['sterling_ratio'], 2)
        sterling_rank_str = f"{data['sterling_rank_position']}th ({sterling_ratio_value})"
        price_drop_rank_str = f"{data['price_drop_rank_position']}th largest"
        
        output_data.append({
            'Stock': symbol,
            'Sterling Ratio Rank': sterling_rank_str,
            'Price Drop Rank': price_drop_rank_str,
            'Alpha Group': data['alpha_group'],
            'Core (%)': data['core_allocation'],
            'Satellite (%)': data['satellite_allocation']
        })
    
    # Create DataFrame and sort by allocation (descending)
    df = pd.DataFrame(output_data)
    df['Total Allocation'] = df['Core (%)'] + df['Satellite (%)']
    df = df.sort_values(by='Total Allocation', ascending=False)
    
    # Reorder columns and drop Total Allocation
    df = df[['Stock', 'Sterling Ratio Rank', 'Price Drop Rank', 'Alpha Group', 'Core (%)', 'Satellite (%)']]
    
    return df

def generate_summary(output_df, stocks_analysis):
    """Generate summary and rationale for allocation decisions."""
    # Count stocks in each portfolio
    core_stocks = output_df[output_df['Core (%)'] > 0]
    satellite_stocks = output_df[output_df['Satellite (%)'] > 0]
    
    # Calculate total allocations
    total_core = core_stocks['Core (%)'].sum()
    total_satellite = satellite_stocks['Satellite (%)'].sum()
    
    # Generate summary
    summary = {
        'core_count': len(core_stocks),
        'satellite_count': len(satellite_stocks),
        'total_core_allocation': total_core,
        'total_satellite_allocation': total_satellite,
        'core_stocks': core_stocks['Stock'].tolist(),
        'satellite_stocks': satellite_stocks['Stock'].tolist()
    }
    
    # Generate rationale
    rationale = """
    Core-Satellite Allocation Rationale:
    
    1. Core Portfolio (70-80% of capital):
       - Selected stocks with higher Sterling ratios relative to the S&P 500 benchmark
       - Sterling ratio measures risk-adjusted return using drawdowns rather than volatility
       - Prioritized stocks trading closer to their 3-month lows
       - Balanced with consideration for potential recovery (price drop metric)
       
    2. Satellite Portfolio (20-30% of capital):
       - Included stocks with lower Sterling ratios
       - Focused on stocks showing significant price drops
       - Selected based on potential for higher returns with managed risk
       
    3. Risk Management:
       - Core portfolio provides stability and steady returns
       - Satellite portfolio offers growth potential
       - Combined approach balances risk and reward
    """
    
    return summary, rationale

def main():
    """Main function to execute the Core-Satellite Capital Allocation process."""
    # Install yfinance if not already installed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        print("yfinance installed successfully")
    
    # Read stocks from CSV
    csv_path = "portfolio_pick.csv"  # Update this path to your CSV file location
    stocks_data = read_stocks_from_csv(csv_path)
    
    # Extract stock symbols
    stock_symbols = [stock['Stock'] for stock in stocks_data]
    
    # Fetch benchmark data (S&P 500)
    print("Fetching benchmark data (S&P 500)...")
    benchmark_symbol = "^GSPC"
    benchmark_data = fetch_stock_data(benchmark_symbol)
    
    if benchmark_data is None:
        print(f"Failed to fetch benchmark data for {benchmark_symbol}. Exiting.")
        return
    
    # Fetch data for each stock
    print(f"Fetching data for {len(stock_symbols)} stocks...")
    stocks_historical_data = {}
    for symbol in stock_symbols:
        print(f"Fetching data for {symbol}...")
        stock_data = fetch_stock_data(symbol)
        if stock_data is not None:
            stocks_historical_data[symbol] = stock_data
    
    # Calculate metrics for each stock
    print("Calculating metrics for each stock...")
    stocks_analysis = {}
    for symbol, stock_data in stocks_historical_data.items():
        # Calculate Sterling ratio metrics against S&P 500 benchmark
        sterling_metrics = calculate_sterling_ratio(stock_data, benchmark_data)
        
        # Calculate price movement metrics
        price_movements = calculate_price_movements(stock_data)
        
        # Store metrics
        stocks_analysis[symbol] = {
            'sterling': sterling_metrics,
            'price_movements': price_movements
        }
    
    # Rank stocks based on criteria
    print("Ranking stocks based on criteria...")
    ranked_stocks = rank_stocks(stocks_analysis)
    
    # Allocate portfolio
    print("Allocating portfolio using Core-Satellite approach...")
    allocated_stocks = allocate_portfolio(ranked_stocks)
    
    # Create output DataFrame
    print("Creating output DataFrame...")
    output_df = create_output_dataframe(allocated_stocks)
    
    # Generate summary and rationale
    print("Generating summary and rationale...")
    summary, rationale = generate_summary(output_df, stocks_analysis)
    
    # Print results
    print("\nCore-Satellite Capital Allocation Results:")
    print(output_df)
    
    # Print summary
    print("\nSummary:")
    print(f"Core Portfolio: {summary['core_count']} stocks, {summary['total_core_allocation']}% allocation")
    print(f"Satellite Portfolio: {summary['satellite_count']} stocks, {summary['total_satellite_allocation']}% allocation")
    
    # Save results to CSV
    output_file = "core_satellite_allocation_sterling_results.csv"
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Return the output DataFrame for further use
    return output_df

if __name__ == "__main__":
    main()
