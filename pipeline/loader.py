"""
Handles data loading, preprocessing, and feature-target construction
for multi-ticker return forecasting tasks.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple


def fetch_price_data(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads adjusted close prices and volumes for the given tickers using yfinance.

    Args:
        tickers (List[str]): List of stock tickers.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Close prices and volume data (dates as index).
    """
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = data["Close"].ffill().bfill().dropna(how="all")
    volumes = data["Volume"].ffill().bfill().dropna(how="all")
    return prices, volumes


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes log returns from price data.

    Args:
        prices (pd.DataFrame): Adjusted close price data.

    Returns:
        pd.DataFrame: Log return series for each ticker.
    """
    return np.log(prices / prices.shift(1)).dropna()


def create_sliding_windows(
    data: pd.DataFrame,
    window_size: int,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts multivariate time series into 3D sliding windows for supervised learning.

    Args:
        data (pd.DataFrame): Log return data with tickers as columns.
        window_size (int): Number of time steps per input sequence.
        horizon (int): Number of steps ahead to predict (usually 1 for next-day return).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (N, T, F) and targets (N,)
    """
    X, y = [], []
    tickers = data.columns
    values = data.values

    for i in range(window_size, len(values) - horizon + 1):
        window = values[i - window_size:i]
        target = values[i + horizon - 1]
        X.append(window)
        y.append(target)

    X = np.array(X)  # shape: (samples, time_steps, tickers)
    y = np.array(y)  # shape: (samples, tickers)

    return X, y

    
def create_sequences(price_df: pd.DataFrame, volume_df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to create input sequences and targets from raw price and volume data.

    Args:
        price_df (pd.DataFrame): DataFrame of adjusted close prices.
        volume_df (pd.DataFrame): DataFrame of trading volumes (unused for now).
        window_size (int): Length of historical window to use.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 3D input sequences and 2D return targets.
    """
    returns = compute_log_returns(price_df)
    X, y = create_sliding_windows(returns, window_size)
    return X, y