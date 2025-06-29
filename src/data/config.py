"""
Configuration settings for financial data pipeline.

All important parameters are defined here for easy modification and maintenance.
"""

# Data configuration
DATA_CONFIG = {
    'expected_columns': ['date', 'vix', 'treasury_13w', 'treasury_10y', 'credit_spread'],
    'date_formats': ['%Y-%m-%d'],
    'missing_value_threshold': 0.1,  # 10% missing values threshold
    'outlier_std_threshold': 3,      # Standard deviations for outlier detection
}

# Financial range validation
FINANCIAL_RANGES = {
    'vix': {
        'absolute_min': 0, 'absolute_max': 200,
        'typical_min': 8, 'typical_max': 80,
        'extreme_min': 5, 'extreme_max': 150,
        'description': 'VIX (Volatility Index)'
    },
    'treasury_13w': {
        'absolute_min': -2, 'absolute_max': 20,
        'typical_min': 0, 'typical_max': 8,
        'extreme_min': -1, 'extreme_max': 15,
        'description': '13-Week Treasury Yield (%)'
    },
    'treasury_10y': {
        'absolute_min': -2, 'absolute_max': 20,
        'typical_min': 0.5, 'typical_max': 8,
        'extreme_min': 0, 'extreme_max': 15,
        'description': '10-Year Treasury Yield (%)'
    },
    'credit_spread': {
        'absolute_min': -100, 'absolute_max': 5000,
        'typical_min': 0, 'typical_max': 1000,
        'extreme_min': -50, 'extreme_max': 2000,
        'description': 'Credit Spread (basis points)'
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'rolling_windows': [5, 10, 20, 50, 100],
    'momentum_periods': [1, 3, 5, 10, 20],
    'correlation_windows': [10, 20, 30, 60],
    'regime_windows': [20, 60, 120],
    'volatility_windows': [5, 10, 20],
}

# Regime identification configuration
REGIME_CONFIG = {
    'vix_thresholds': {
        'low': 20,      # VIX < 20 = low volatility
        'high': 30,     # VIX > 30 = high volatility
        'extreme': 40   # VIX > 40 = extreme volatility
    },
    'smoothing_window': 3,  # Days for regime label smoothing
    'min_regime_duration': 2,  # Minimum days for a regime
}

# Historical crisis periods
HISTORICAL_CRISES = [
    ('2018-02-01', '2018-03-31', 'Volatility Spike'),
    ('2018-10-01', '2018-12-31', 'Q4 2018 Selloff'),
    ('2020-02-01', '2020-05-31', 'COVID-19 Crisis'),
    ('2022-02-01', '2022-04-30', 'Russia-Ukraine War'),
    ('2023-03-01', '2023-03-31', 'Banking Crisis (SVB)')
]

# Train/validation/test split configuration
SPLIT_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'temporal_split': True,  # Maintain temporal order
    'ensure_all_regimes': True,  # Ensure all regimes in train set
}

# Output configuration
OUTPUT_CONFIG = {
    'save_intermediate': True,
    'create_visualizations': True,
    'generate_reports': True,
    'file_formats': ['csv'],  # Supported file formats
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'save_format': 'png',
}


