"""
Utility functions for the financial data pipeline
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import warnings

def validate_file_path(file_path):
    """Validate that file exists and is readable"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File not readable: {file_path}")
    
    return True

def create_directory_structure(base_dir):
    """Create the complete directory structure for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'reports',
        'visualizations',
        'models',  # For Person 2
        'logs'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
    
    print(f"✓ Created directory structure in {base_dir}")

def safe_divide(numerator, denominator, fill_value=np.nan):
    """Safely divide two series, handling division by zero"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], fill_value)
    return result

def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    """Winsorize a series to handle extreme outliers"""
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    
    return series.clip(lower=lower_bound, upper=upper_bound)

def calculate_rolling_quantile(series, window, quantile, min_periods=None):
    """Calculate rolling quantile with proper error handling"""
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    return series.rolling(window=window, min_periods=min_periods).quantile(quantile)

def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def business_days_between(start_date, end_date):
    """Calculate number of business days between two dates"""
    return pd.bdate_range(start=start_date, end=end_date).shape[0]

def get_memory_usage(df):
    """Get memory usage of DataFrame in MB"""
    return df.memory_usage(deep=True).sum() / 1024 / 1024

def log_execution_time(func):
    """Decorator to log execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"✓ {func.__name__} completed in {execution_time:.2f} seconds")
        return result
    return wrapper

def save_json_safely(data, file_path):
    """Save data to JSON with proper error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert any non-serializable objects
        serializable_data = make_json_serializable(data)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        print(f"✓ Saved JSON: {file_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to save JSON {file_path}: {e}")
        return False

def make_json_serializable(obj):
    """Convert pandas/numpy objects to JSON serializable formats"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

def validate_data_quality(df, required_columns=None, missing_threshold=0.1):
    """Validate basic data quality requirements"""
    issues = []
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check missing values
    missing_pct = df.isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > missing_threshold]
    if len(high_missing) > 0:
        issues.append(f"High missing values: {high_missing.to_dict()}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    return issues

def create_feature_importance_report(feature_names, importance_scores=None):
    """Create a feature importance report (placeholder for future ML integration)"""
    # 🚨 NOTE: This would typically use ML model feature importance
    # For now, create based on variance as a proxy
    report = {
        'total_features': len(feature_names),
        'feature_categories': {},
        'top_features_by_variance': [],  # Would be filled with actual importance
        'feature_descriptions': {}
    }
    
    # Categorize features
    categories = ['return', 'vol', 'momentum', 'regime', 'yield', 'vix', 'corr']
    for category in categories:
        count = len([f for f in feature_names if category in f.lower()])
        if count > 0:
            report['feature_categories'][category] = count
    
    return report

def generate_data_summary_stats(df, financial_cols=None):
    """Generate comprehensive summary statistics"""
    if financial_cols is None:
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
    
    available_cols = [col for col in financial_cols if col in df.columns]
    
    summary = {
        'shape': df.shape,
        'date_range': {
            'start': str(df['date'].min()) if 'date' in df.columns else None,
            'end': str(df['date'].max()) if 'date' in df.columns else None,
            'total_days': len(df)
        },
        'descriptive_stats': {},
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Calculate descriptive statistics for financial columns
    if available_cols:
        desc_stats = df[available_cols].describe()
        summary['descriptive_stats'] = desc_stats.to_dict()
    
    return summary

