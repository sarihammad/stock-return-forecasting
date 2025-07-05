"""
Defines baseline models for return forecasting: Linear Regression and XGBoost.
"""

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def get_linear_model():
    """
    Returns a Linear Regression model.

    Returns:
        sklearn.linear_model.LinearRegression: Linear model instance.
    """
    return LinearRegression()


def get_xgb_model():
    """
    Returns a default XGBoost Regressor with preset hyperparameters.

    Returns:
        xgboost.XGBRegressor: XGBoost model instance.
    """
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )