#!/usr/bin/env python3
"""
step 2 data extraction
Step 0: Download financial data from yfinance
Usage: python 01_download_data.py
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
#add tickers to this to export more than one at a time
# Configuration
TICKERS = ["SPY"]  # Add more: ["SPY", "QQQ", "AAPL", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"  # or use datetime.now().strftime("%Y-%m-%d") for today
INTERVAL = "1d"  # Options: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo

# Folders
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    print("✓ Directory structure created")

def download_ticker(ticker, start, end, interval):
    """
    Download data for a single ticker with error handling
    
    Args:
        ticker: Stock symbol (e.g., "SPY")
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: Data interval (1d, 1h, etc.)
    
    Returns:
        DataFrame or None if failed
    """
    try:
        print(f"\nDownloading {ticker}...")
        data = yf.download(
            ticker, 
            start=start, 
            end=end, 
            interval=interval,
            progress=False  # Disable progress bar for cleaner output
        )
        
        if data.empty:
            print(f"✗ No data returned for {ticker}")
            return None
        
        # Add ticker column for multi-ticker analysis later
        data['Ticker'] = ticker
        
        print(f"✓ Downloaded {len(data)} rows for {ticker}")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        
        return data
    
    except Exception as e:
        print(f"✗ Error downloading {ticker}: {e}")
        return None

def save_data(data, ticker, interval, start_date, end_date):
    """Save data to CSV with proper naming including date range"""
    # Convert dates to clean format (remove dashes)
    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")
    filename = f"{ticker.lower()}_{interval}_{start_str}_{end_str}.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    data.to_csv(filepath)
    print(f"✓ Saved to {filepath}")
    
    return filepath

def data_quality_check(data, ticker):
    """Quick quality check on downloaded data"""
    print(f"\n--- Data Quality Check: {ticker} ---")
    print(f"Shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # Check for suspicious values
    if 'Close' in data.columns:
        close_prices = data['Close']
        print(f"\nPrice stats:")
        print(f"  Min: ${close_prices.min():.2f}")
        print(f"  Max: ${close_prices.max():.2f}")
        print(f"  Mean: ${close_prices.mean():.2f}")
        
        # Check for zeros or negative prices
        if (close_prices <= 0).any():
            print("⚠ WARNING: Found zero or negative prices!")
    
    # Show last few rows
    print(f"\nLast 5 rows:")
    print(data.tail())

def main():
    print("=" * 60)
    print("FINANCIAL DATA DOWNLOADER")
    print("=" * 60)
    
    # Setup
    create_directories()
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Interval: {INTERVAL}")
    
    # Download all tickers
    all_data = {}
    successful = []
    failed = []
    
    for ticker in TICKERS:
        data = download_ticker(ticker, START_DATE, END_DATE, INTERVAL)
        
        if data is not None:
            all_data[ticker] = data
            filepath = save_data(data, ticker, INTERVAL, START_DATE, END_DATE)
            data_quality_check(data, ticker)
            successful.append(ticker)
        else:
            failed.append(ticker)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Successful: {len(successful)} ({', '.join(successful)})")
    if failed:
        print(f"✗ Failed: {len(failed)} ({', '.join(failed)})")
    '''
    # Create combined dataset if multiple tickers
    if len(all_data) > 1:
        print("\nCreating combined dataset...")
        combined = pd.concat(all_data.values())
        start_str = START_DATE.replace("-", "")
        end_str = END_DATE.replace("-", "")
        combined_path = os.path.join(RAW_DATA_DIR, f"combined_{INTERVAL}_{start_str}_{end_str}.csv")
        combined.to_csv(combined_path)
        print(f"✓ Saved combined data to {combined_path}")
    
    print("\n" + "=" * 60)
    print("✓ DATA DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nNext step: Run exploratory data analysis (EDA)")
    print(f"Saved files in: {RAW_DATA_DIR}/")
    '''
if __name__ == '__main__':
    main()
