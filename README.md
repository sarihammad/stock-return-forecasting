# LSTM Forecasting for Multi-Ticker Returns

Compares three models (LSTM, Linear Regression, and XGBoost) to forecast next-day log returns for a group of major U.S. stocks using historical price data from Yahoo Finance.

## Features

The project uses PyTorch to train a multi-output LSTM model and Scikit-learn to implement baseline models, including Linear Regression and XGBoost. It evaluates predictive performance using metrics like mean squared error (MSE), Spearman correlation (information coefficient), and directional accuracy.


## Model Evaluation

Using 11 tickers from 2015–2024:

| Model              | MSE       | Information Coefficient (IC) | Directional Accuracy |
|-------------------|-----------|-------------------------------|----------------------|
| **LSTM**          | 0.000415  | 0.0097                        | 52.43%               |
| **Linear Regression** | 0.000492  | 0.0105                        | 50.37%               |
| **XGBoost**       | 0.000429  | 0.0053                        | 50.96%               |

> The LSTM model slightly outperformed the baselines, especially in directional accuracy — a key signal in financial forecasting.

## How to Run

```bash
pip install -r requirements.txt
python main.py
```
