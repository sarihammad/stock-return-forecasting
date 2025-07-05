"""
Handles training of models (LSTM, Linear Regression, XGBoost) 
on multi-ticker return forecasting datasets.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from model.baselines import get_linear_model, get_xgb_model


def train_lstm_model(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    verbose: bool = True
) -> nn.Module:
    """
    Trains an LSTM model using MSE loss.

    Args:
        model (nn.Module): Initialized LSTM model.
        train_data (Tuple): Tuple of (X_train, y_train) tensors.
        val_data (Tuple): Tuple of (X_val, y_val) tensors.
        num_epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        verbose (bool): Whether to print loss during training.

    Returns:
        nn.Module: Trained model.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    X_train, y_train = train_data
    X_val, y_val = val_data

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if verbose:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val).squeeze()
                val_loss = loss_fn(val_preds, y_val).item()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model


def train_linear_model(X_train, y_train):
    """
    Trains a Linear Regression model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.

    Returns:
        sklearn.linear_model.LinearRegression: Trained model.
    """
    model = get_linear_model()
    model.fit(X_train, y_train)
    return model


def train_xgb_model(X_train, y_train):
    """
    Trains an XGBoost Regressor.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.

    Returns:
        xgboost.XGBRegressor: Trained model.
    """
    model = get_xgb_model()
    model.fit(X_train, y_train)
    return model