import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler


def create_supervised_dataset(
    returns: pd.DataFrame,
    lookback: int = 10,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Converts a return DataFrame into a supervised learning format.

    Args:
        returns (pd.DataFrame): Log return DataFrame (dates as index, tickers as columns).
        lookback (int): Number of days of past returns as features.
        horizon (int): Days ahead to predict.

    Returns:
        X (np.ndarray): Features of shape (samples, lookback, num_tickers).
        y (np.ndarray): Targets of shape (samples, num_tickers).
        dates (List[str]): Dates corresponding to each sample.
        tickers (List[str]): List of tickers.
    """
    tickers = returns.columns.tolist()
    data = returns.values

    X, y, dates = [], [], []
    for i in range(lookback, len(data) - horizon):
        X.append(data[i - lookback:i])
        y.append(data[i + horizon])
        dates.append(returns.index[i + horizon])

    X = np.array(X)
    y = np.array(y)
    return X, y, dates, tickers


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scales feature arrays using standard normalization.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.

    Returns:
        X_train_scaled (np.ndarray): Scaled training features.
        X_test_scaled (np.ndarray): Scaled testing features.
        scaler (StandardScaler): Fitted scaler instance.
    """
    scaler = StandardScaler()
    num_samples, lookback, num_assets = X_train.shape

    # Flatten across time and asset dim, then reshape back
    X_train_2d = X_train.reshape(num_samples, -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(num_samples, lookback, num_assets)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape[0], lookback, num_assets)

    return X_train_scaled, X_test_scaled, scaler