# src/models/utils.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data with windowing"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 sequence_length: int, prediction_horizon: int = 1):
        
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        #windowing indexes
        self.indices = list(range(sequence_length, len(data) - prediction_horizon + 1))
    
    def __len__(self):
        return len(self.indices)
        
    
    def __getitem__(self, idx):
        
        #end index 
        end_idx = self.indices[idx]
        #start of sequence
        start_idx = end_idx - self.sequence_length
        #target index
        target_idx = end_idx + self.prediction_horizon - 1
  
        sequence = self.data[start_idx:end_idx]
        target = self.targets[target_idx]
        #convert to tensors
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


def create_sequences(data: np.ndarray, sequence_length: int, 
                    prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    #creates input sequences and targets for LSTM training
    x, y = [], [] 
    
    
    #looping through valid indexes
    for i in range(sequence_length, len(data) - prediction_horizon + 1):
        #input sequence
        x.append(data[i-sequence_length:i])
        #target
        y.append(data[i + prediction_horizon - 1])
        
    return np.array(x), np.array(y)


def train_test_split(data: pd.DataFrame, 
                    train_ratio: float = 0.70,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    target_columns: Optional[List[str]] = None) -> Tuple[dict, dict]:
    
    
    if target_columns is None:
        target_columns = data.columns[-4:].tolist()
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    #calculating sizes from ratios
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size  
    
    print(f"Split sizes: Train={train_size}, Val={val_size}, Test={test_size}")

    #split data
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size] 
    test_data = data.iloc[train_size + val_size:train_size + val_size + test_size]
    
    #separates features from targets
    feature_columns = [col for col in data.columns if col not in target_columns]
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()


    
    train_features_scaled = feature_scaler.fit_transform(train_data[feature_columns])
    train_targets_scaled = target_scaler.fit_transform(train_data[target_columns])
    
    
    val_features_scaled = feature_scaler.transform(val_data[feature_columns])
    val_targets_scaled = target_scaler.transform(val_data[target_columns])

    test_features_scaled = feature_scaler.transform(test_data[feature_columns])
    test_targets_scaled = target_scaler.transform(test_data[target_columns])

    #dicts with splits and scaled data
    datasets = {
        'train': {
            'features': train_features_scaled,           # Scaled features for training
            'targets': train_targets_scaled,             # Scaled targets for training  
            'raw_features': train_data[feature_columns].values,  # Original features
            'raw_targets': train_data[target_columns].values     # Original targets
        },
        'val': {
            'features': val_features_scaled,
            'targets': val_targets_scaled,
            'raw_features': val_data[feature_columns].values,
            'raw_targets': val_data[target_columns].values
        },
        'test': {
            'features': test_features_scaled,
            'targets': test_targets_scaled,
            'raw_features': test_data[feature_columns].values,
            'raw_targets': test_data[target_columns].values
        }
    }
    
    # dictionary with fitted scalers
    scalers = {
        'feature_scaler': feature_scaler, 
        'target_scaler': target_scaler
    }
    
    return (datasets, scalers)

def create_data_loaders(datasets: dict, sequence_length: int, 
                       batch_size: int = 32, prediction_horizon: int = 1) -> dict:
    
    loaders = {}
    
    
    for split_name, split_data in datasets.items():
        
        dataset = TimeSeriesDataset(
            data = split_data['features'],
            targets = split_data['targets'],
            sequence_length = sequence_length,
            prediction_horizon = prediction_horizon
        )
        #shuffle only for training
        shuffle = (split_name == 'train')
        
        loaders[split_name] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=False
        )
        
    
    return loaders


