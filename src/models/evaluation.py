# src/models/evaluation.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    """Comprehensive evaluation metrics for time series forecasting"""
    
    def __init__(self, target_names: Optional[List[str]] = None):
        self.target_names = target_names or [f'target_{i}' for i in range(4)]
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         regime_labels: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        HINT:
        - Create empty metrics dictionary
        - Calculate overall metrics: MSE, RMSE, MAE, MAPE
        - Calculate per-variable metrics in a loop
        - Calculate regime-specific metrics if regime_labels provided
        - Use sklearn functions: mean_squared_error(), mean_absolute_error()
        - RMSE = sqrt(MSE)
        """
        #calculate and add metrics to dict
        metrics = {}
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mape'] = self._calculate_mape(y_true, y_pred)
        
        
       #per var metrics
        metrics['per_variable'] = {}


        for i, var_name in enumerate(self.target_names):
            var_metrics = {
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'mape': self._calculate_mape(y_true[:, i], y_pred[:, i]),
                'directional_accuracy': self._directional_accuracy(y_true[:, i], y_pred[:, i])
            }
            var_metrics['rmse'] = np.sqrt(var_metrics['mse'])
            metrics['per_variable'][var_name] = var_metrics
        

        metrics['directional_accuracy'] = np.mean([
            metrics['per_variable'][var]['directional_accuracy'] 
            for var in self.target_names
        ])
        
        if regime_labels is not None:
            metrics['regime_specific'] = self._calculate_regime_metrics(
                y_true, y_pred, regime_labels
            )
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error
        
        HINT:
        - Create mask to avoid division by zero: mask = np.abs(y_true) > 1e-8
        - If no valid values, return 0.0
        - Calculate: mean(abs((y_true - y_pred) / y_true)) * 100
        - Only use values where mask is True
        """
        mask = np.abs(y_true) > 1e-8
        if not np.any(mask):
            return 0.0

        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return 0.0
            
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction)
    
    def _calculate_regime_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 regime_labels: np.ndarray) -> Dict:
        regime_metrics = {}
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_true = y_true[regime_mask]
            regime_pred = y_pred[regime_mask]
            
            if len(regime_true) > 0:
                regime_metrics[f'regime_{regime}'] = {
                    'mse': mean_squared_error(regime_true, regime_pred),
                    'mae': mean_absolute_error(regime_true, regime_pred),
                    'samples': len(regime_true)
                }
        
        return regime_metrics
    
    def print_metrics(self, metrics: Dict, title: str = "Model Performance"):
        
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        
        # Overall metrics
        print(f"Overall Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        # Per-variable metrics
        print(f"\nPer-Variable Metrics:")
        for var_name, var_metrics in metrics['per_variable'].items():
            print(f"  {var_name}:")
            print(f"    RMSE: {var_metrics['rmse']:.4f}")
            print(f"    MAE:  {var_metrics['mae']:.4f}")
            print(f"    Dir. Acc: {var_metrics['directional_accuracy']:.2f}%")


        if 'regime_specific' in metrics:
            print(f"\nRegime-Specific Metrics:")
            for regime_name, regime_metrics in metrics['regime_specific'].items():
                print(f"  {regime_name}:")
                print(f"    RMSE: {np.sqrt(regime_metrics['mse']):.4f}")
                print(f"    MAE:  {regime_metrics['mae']:.4f}")
                print(f"    Samples: {regime_metrics['samples']}")


def train_model(model: torch.nn.Module, train_loader: DataLoader, 
               val_loader: DataLoader, num_epochs: int = 100,
               learning_rate: float = 0.001, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    history = {'train_loss': train_losses, 'val_loss': val_losses}
    return model, history


"""
# Example usage script
if __name__ == "__main__":
    # Example of how to use the framework
    import pandas as pd
    
    # Load your data
    # data = pd.read_csv('your_timeseries_data.csv')
    
    # For demonstration, create synthetic data
    np.random.seed(42)
    n_samples = 2000
    n_features = 10
    
    # Create synthetic time series data
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features-4)] + 
                [f'target_{i}' for i in range(4)]
    )
    
    # Define target columns
    target_columns = [f'target_{i}' for i in range(4)]
    
    # Split data
    datasets, scalers = train_test_split(data, target_columns=target_columns)
    
    # Create data loaders
    sequence_length = 20
    batch_size = 32
    loaders = create_data_loaders(datasets, sequence_length, batch_size)
    
    # Initialize model
    input_size = datasets['train']['features'].shape[1]
    model = BaseLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model, history = train_model(model, loaders['train'], loaders['val'], 
                                       num_epochs=50, device=device)
    
    # Evaluate model
    evaluator = ModelEvaluator(target_names=target_columns)
    
    # Get predictions on test set
    trained_model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in loaders['test']:
            batch_x = batch_x.to(device)
            outputs = trained_model(batch_x)
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(batch_y.numpy())
    
    # Combine predictions
    test_predictions = np.vstack(test_predictions)
    test_targets = np.vstack(test_targets)
    
    # Calculate and print metrics
    metrics = evaluator.calculate_metrics(test_targets, test_predictions)
    evaluator.print_metrics(metrics, "LSTM Baseline Performance")

"""