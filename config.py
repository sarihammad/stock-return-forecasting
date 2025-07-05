"""
Configuration settings for LSTM-based return forecasting.

Includes:
- Ticker universe
- Date range
- Feature and model hyperparameters
"""

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "UNH", "SPY"
]

# date range for historical data
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# feature engineering
SEQ_LENGTH = 20  # number of days in each input sequence
FEATURES = ["return", "volume_change"]  

# model hyperparameters
INPUT_SIZE = len(TICKERS)         # number of input features per timestep
HIDDEN_SIZE = 64                   # hidden units in LSTM
NUM_LAYERS = 2                     # stacked LSTM layers
DROPOUT = 0.2                      # dropout between LSTM layers

# training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# evaluation
TARGET_HORIZON = 1  # predict next-day return
TEST_SPLIT_RATIO = 0.2  # proportion of data reserved for testing