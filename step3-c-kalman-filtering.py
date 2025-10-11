#!/usr/bin/env python3
"""
step 3 kalman filtering
PROCESS_ALL_TICKERS = True,USE_MOST_RECENT = True  for using all most recent tickers
PROCESS_ALL_TICKERS = False ,USE_MOST_RECENT = False for using selected, and oldest  
Step 1: Kalman Filtering for noise reduction
Usage: python 02_kalman_filtering.py
"""

import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import os
import glob

# Configuration
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUTS_DIR = "outputs"

# Ticker selection
# Option 1: Process ALL tickers found in raw data
PROCESS_ALL_TICKERS = True

# Option 2: Process only specific tickers (set PROCESS_ALL_TICKERS = False)
SPECIFIC_TICKERS = ["SPY", "QQQ", "AAPL"]  # Only process these

# Date selection for each ticker
USE_MOST_RECENT = True  # True = use latest file, False = use oldest file

# Kalman Filter parameters
KALMAN_PARAMS = {
    'transition_matrices': [1],
    'observation_matrices': [1],
    'observation_covariance': 1,      # Measurement noise
    'transition_covariance': 0.01,    # Process noise (lower = smoother)
}

def get_latest_file(ticker, data_dir=RAW_DATA_DIR, use_most_recent=True):
    """
    Find the most recent (or oldest) file for a given ticker
    
    Args:
        ticker: Stock symbol (e.g., "SPY")
        data_dir: Directory to search in
        use_most_recent: True for latest file, False for oldest
    
    Returns:
        Path to file or None
    """
    pattern = os.path.join(data_dir, f"{ticker.lower()}_*.csv")
    files = glob.glob(pattern)
    
    # Exclude combined files
    files = [f for f in files if 'combined' not in os.path.basename(f).lower()]
    
    if not files:
        return None
    
    # Sort by filename (date is in filename, so this works)
    files_sorted = sorted(files)
    
    if use_most_recent:
        return files_sorted[-1]  # Last = most recent
    else:
        return files_sorted[0]   # First = oldest

def apply_kalman_filter(data, column='Close', params=None):
    """
    Apply Kalman filter to a price series
    
    Args:
        data: DataFrame with price data
        column: Column to filter
        params: Kalman filter parameters (uses defaults if None)
    
    Returns:
        Filtered values, covariances
    """
    if params is None:
        params = KALMAN_PARAMS.copy()
    
    # Set initial state to first price
    params['initial_state_mean'] = data[column].iloc[0]
    params['initial_state_covariance'] = 1
    
    # Initialize Kalman Filter
    kf = KalmanFilter(**params)
    
    # Smooth the data (forward-backward pass for best estimates)
    state_means, state_covariances = kf.smooth(data[column].values)
    
    return state_means.flatten(), state_covariances.flatten()

def calculate_noise_metrics(original, filtered):
    """Calculate noise reduction metrics"""
    noise = original - filtered
    
    metrics = {
        'noise_std': np.std(noise),
        'noise_mean': np.mean(noise),
        'snr_improvement': np.std(original) / np.std(noise),
        'max_deviation': np.max(np.abs(noise))
    }
    
    return metrics

def process_ticker(ticker, raw_data_dir=RAW_DATA_DIR, 
                   processed_data_dir=PROCESSED_DATA_DIR,
                   outputs_dir=OUTPUTS_DIR):
    """
    Process a single ticker through Kalman filtering
    
    Args:
        ticker: Stock symbol
        raw_data_dir: Directory with raw data
        processed_data_dir: Directory for processed output
        outputs_dir: Directory for plots
    
    Returns:
        DataFrame with filtered data or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print(f"{'='*60}")
    
    # Find file (most recent or oldest based on config)
    filepath = get_latest_file(ticker, raw_data_dir, USE_MOST_RECENT)
    if filepath is None:
        print(f"✗ No data file found for {ticker}")
        return None
    
    file_type = "most recent" if USE_MOST_RECENT else "oldest"
    print(f"✓ Loading {file_type} file: {os.path.basename(filepath)}")
    
    # Load data
    try:
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    except Exception as e:
        print(f"✗ Error loading {ticker}: {e}")
        return None
    
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Apply Kalman filter to Close price
    print("\nApplying Kalman filter...")
    close_filtered, close_cov = apply_kalman_filter(df, column='Close')
    df['Close_KF'] = close_filtered
    df['Close_KF_Variance'] = close_cov
    
    # Calculate returns (both original and filtered)
    df['Returns'] = df['Close'].pct_change()
    df['Returns_KF'] = df['Close_KF'].pct_change()
    
    # Also filter High and Low for complete OHLC
    print("Filtering OHLC data...")
    for col in ['Open', 'High', 'Low']:
        if col in df.columns:
            filtered, _ = apply_kalman_filter(df, column=col)
            df[f'{col}_KF'] = filtered
    
    # Calculate noise metrics
    metrics = calculate_noise_metrics(df['Close'].values, df['Close_KF'].values)
    print(f"\n--- Noise Reduction Metrics ---")
    print(f"  Noise std: {metrics['noise_std']:.4f}")
    print(f"  SNR improvement: {metrics['snr_improvement']:.2f}x")
    print(f"  Max deviation: ${metrics['max_deviation']:.2f}")
    
    # Save processed data
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Extract date range from filename
    basename = os.path.basename(filepath)
    parts = basename.replace('.csv', '').split('_')
    if len(parts) >= 4:
        # Format: ticker_interval_start_end
        output_filename = f"{parts[0]}_kalman_{parts[2]}_{parts[3]}.csv"
    else:
        output_filename = f"{ticker.lower()}_kalman.csv"
    
    output_path = os.path.join(processed_data_dir, output_filename)
    df.to_csv(output_path)
    print(f"\n✓ Saved processed data to: {output_path}")
    
    # Create visualization
    plot_kalman_comparison(df, ticker, outputs_dir, output_filename.replace('.csv', '.png'))
    
    return df

def plot_kalman_comparison(df, ticker, outputs_dir, filename):
    """Create comparison plots of original vs filtered data"""
    os.makedirs(outputs_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Price comparison
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Original Close', alpha=0.6, linewidth=1)
    ax1.plot(df.index, df['Close_KF'], label='Kalman Filtered', linewidth=2, color='red')
    ax1.set_title(f'{ticker} - Price: Original vs Kalman Filtered', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Noise removed
    ax2 = axes[1]
    noise = df['Close'] - df['Close_KF']
    ax2.plot(df.index, noise, color='gray', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(df.index, noise, 0, alpha=0.3, color='gray')
    ax2.set_title('Noise Removed by Kalman Filter', fontsize=12)
    ax2.set_ylabel('Noise ($)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Returns comparison
    ax3 = axes[2]
    ax3.plot(df.index, df['Returns'], label='Original Returns', alpha=0.5, linewidth=1)
    ax3.plot(df.index, df['Returns_KF'], label='Filtered Returns', linewidth=1.5, color='green')
    ax3.set_title('Returns: Original vs Filtered', fontsize=12)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Returns', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    output_path = os.path.join(outputs_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("KALMAN FILTERING - NOISE REDUCTION")
    print("=" * 60)
    
    # Check for raw data files
    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not raw_files:
        print(f"✗ No data files found in {RAW_DATA_DIR}")
        print("Run 01_download_data.py first!")
        return
    
    print(f"\nFound {len(raw_files)} data file(s)")
    
    # Determine which tickers to process
    if PROCESS_ALL_TICKERS:
        # Extract unique tickers from filenames
        tickers = set()
        for f in raw_files:
            basename = os.path.basename(f)
            if not basename.startswith('combined'):
                ticker = basename.split('_')[0].upper()
                tickers.add(ticker)
        print(f"Mode: Processing ALL tickers found")
    else:
        # Use only specified tickers
        tickers = set(SPECIFIC_TICKERS)
        print(f"Mode: Processing SPECIFIC tickers only")
    
    print(f"Tickers to process: {', '.join(sorted(tickers))}")
    print(f"File selection: {'Most recent' if USE_MOST_RECENT else 'Oldest'}")
    
    # Show available files for each ticker
    print("\nAvailable files:")
    for ticker in sorted(tickers):
        pattern = os.path.join(RAW_DATA_DIR, f"{ticker.lower()}_*.csv")
        files = [f for f in glob.glob(pattern) if 'combined' not in f.lower()]
        if files:
            print(f"  {ticker}: {len(files)} file(s)")
            for f in sorted(files):
                marker = " ← SELECTED" if f == get_latest_file(ticker, RAW_DATA_DIR, USE_MOST_RECENT) else ""
                print(f"    - {os.path.basename(f)}{marker}")
    
    # Process each ticker
    results = {}
    for ticker in sorted(tickers):
        df = process_ticker(ticker)
        if df is not None:
            results[ticker] = df
    
    # Summary
    print("\n" + "=" * 60)
    print("KALMAN FILTERING SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully processed: {len(results)} ticker(s)")
    if results:
        print(f"\nProcessed files saved in: {PROCESSED_DATA_DIR}/")
        print(f"Plots saved in: {OUTPUTS_DIR}/")
    
    print("\n" + "=" * 60)
    print("✓ KALMAN FILTERING COMPLETE!")
    print("=" * 60)
    print("\nNext step: GARCH volatility modeling (Step 2)")

if __name__ == '__main__':
    main()
