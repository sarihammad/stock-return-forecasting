"""
Main script for training and evaluating return forecasting models (LSTM, Linear, XGBoost).

Steps:
1. Load price and volume data
2. Generate sequences and targets
3. Train LSTM model
4. Train baseline models
5. Evaluate all models
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from pipeline.loader import fetch_price_data, create_sequences
from pipeline.trainer import train_lstm_model, train_linear_model, train_xgb_model
from pipeline.evaluator import evaluate_mse, evaluate_ic, evaluate_directional_accuracy, evaluate_baseline_model
from model.lstm import LSTMModel
from model.baselines import get_linear_model, get_xgb_model
import config

# load data
print("Loading historical price data...")
price_df, volume_df = fetch_price_data(config.TICKERS, config.START_DATE, config.END_DATE)

# create sequences
print("Preparing sequences...")
X, y = create_sequences(price_df, volume_df, config.SEQ_LENGTH)

# train/test split
split_idx = int(len(X) * (1 - config.TEST_SPLIT_RATIO))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# LSTM training
train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float()), batch_size=config.BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMModel(
    input_size=X.shape[2],
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    output_size=X.shape[2],
).to(device)

print("Training LSTM model...")
train_lstm_model(
    lstm_model,
    train_data=(torch.tensor(X_train).float(), torch.tensor(y_train).float()),
    val_data=(torch.tensor(X_test).float(), torch.tensor(y_test).float()),
    num_epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    verbose=True
)

# evaluate LSTM
lstm_model.eval()
with torch.no_grad():
    y_pred_lstm = torch.cat([lstm_model(x.to(device)).squeeze().cpu() for x, _ in test_loader])
    y_true_lstm = torch.cat([y.cpu() for _, y in test_loader])

ic = evaluate_ic(y_true_lstm.numpy(), y_pred_lstm.numpy())
print("LSTM Evaluation:")
print(f"MSE: {evaluate_mse(y_true_lstm, y_pred_lstm):.6f}")
print(f"IC: {ic:.4f}")
print(f"Directional Accuracy: {evaluate_directional_accuracy(y_true_lstm.numpy(), y_pred_lstm.numpy()):.2%}")

# baseline models
X_flat = X.reshape(X.shape[0], -1)
X_train_flat, X_test_flat = X_flat[:split_idx], X_flat[split_idx:]
y_train_vec, y_test_vec = y[:split_idx], y[split_idx:]

# linear regression
lin_model = train_linear_model(X_train_flat, y_train_vec)
mse_lin, ic_lin, acc_lin = evaluate_baseline_model(lin_model, X_test_flat, y_test_vec)
print("Linear Regression Evaluation:")
print(f"MSE: {mse_lin:.6f}")
print(f"IC: {ic_lin:.4f}")
print(f"Directional Accuracy: {acc_lin:.2%}")

# XGBoost
xgb_model = train_xgb_model(X_train_flat, y_train_vec)
mse_xgb, ic_xgb, acc_xgb = evaluate_baseline_model(xgb_model, X_test_flat, y_test_vec)
print("XGBoost Evaluation:")
print(f"MSE: {mse_xgb:.6f}")
print(f"IC: {ic_xgb:.4f}")
print(f"Directional Accuracy: {acc_xgb:.2%}")