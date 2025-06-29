# src/models/base_lstm.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class BaseLSTM(nn.Module):
    """Standard LSTM implementation for multivariate time series forecasting"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 4,
                 dropout: float = 0.2,
                 batch_first: bool = True):
        
        super(BaseLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers

        #basic LSTM creation
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        
        #pass input into LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        if self.batch_first:
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[-1, :, :]
            
        #applying dropout
        output = self.dropout(last_output)

        #getting predictions
        predictions = self.linear(output)

        return predictions
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        
        #creating hidden states
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        return (h_0, c_0)
      


