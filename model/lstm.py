"""
Defines an LSTM model for multi-ticker return forecasting.
"""

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): Number of input features per timestep.
            hidden_size (int): Number of hidden units in LSTM.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability.
            output_size (int): Number of outputs (e.g., number of tickers).
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  

    def forward(self, x):
        """
        Forward pass through LSTM.

        Args:
            x (Tensor): Shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor: Shape (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)  
        last_hidden = lstm_out[:, -1, :] 
        return self.fc(last_hidden) 