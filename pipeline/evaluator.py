"""
Provides evaluation utilities for model performance on return prediction.

Includes:
- Mean Squared Error (MSE)
- Information Coefficient (Spearman)
- Directional Accuracy
"""

import torch
import numpy as np
from scipy.stats import spearmanr

def evaluate_mse(y_true, y_pred) -> float:
    """
    Computes Mean Squared Error between predicted and true returns.

    Args:
        y_true (Union[Tensor, np.ndarray]): Ground truth returns.
        y_pred (Union[Tensor, np.ndarray]): Predicted returns.

    Returns:
        float: MSE value.
    """
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    return torch.mean((y_true - y_pred) ** 2).item()


def evaluate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Spearman rank correlation (Information Coefficient) 
    between predicted and actual returns.

    Args:
        y_true (np.ndarray): True returns.
        y_pred (np.ndarray): Predicted returns.

    Returns:
        float: Spearman correlation coefficient.
    """
    ic, _ = spearmanr(y_true.flatten(), y_pred.flatten())
    return ic if np.isscalar(ic) else float(np.array(ic).flatten()[0])


def evaluate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Evaluates how often the model predicts the correct return direction.

    Args:
        y_true (np.ndarray): Ground truth returns.
        y_pred (np.ndarray): Predicted returns.

    Returns:
        float: Accuracy in predicting up/down movement.
    """
    correct = np.sign(y_true) == np.sign(y_pred)
    return np.mean(correct)


def evaluate_baseline_model(model, X_test, y_test):
    """
    Evaluates a trained baseline model (Linear or XGBoost) on test data.

    Args:
        model: Trained scikit-learn model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.

    Returns:
        Tuple[float, float, float]: MSE, IC, and directional accuracy.
    """
    

    y_pred = model.predict(X_test)

    mse = evaluate_mse(y_test, y_pred)
    ic = evaluate_ic(y_test, y_pred)
    acc = evaluate_directional_accuracy(y_test, y_pred)

    return mse, ic, acc