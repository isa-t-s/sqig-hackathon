import pandas as pd
import numpy as np
from scipy import stats
import warnings

class BasicFeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)
        self.feature_descriptions = {}
    
    def create_basic_returns(self):
        """
        Calculate returns and changes for all variables
        """
        print("Creating basic return features...")
        
        # Define the financial columns
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        for col in available_cols:
            # Daily returns (percentage change)
            self.features[f'{col}_return'] = self.data[col].pct_change()
            self.feature_descriptions[f'{col}_return'] = f'Daily percentage return for {col}'
            
            # Log returns (more suitable for volatility analysis)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.features[f'{col}_log_return'] = np.log(self.data[col] / self.data[col].shift(1))
            self.feature_descriptions[f'{col}_log_return'] = f'Daily log return for {col}'
            
            # First differences (absolute changes)
            self.features[f'{col}_diff'] = self.data[col].diff()
            self.feature_descriptions[f'{col}_diff'] = f'Daily absolute change for {col}'
            
            # Rolling volatility measures
            for window in [5, 10, 20]:
                # Volatility of returns
                self.features[f'{col}_vol_{window}d'] = (
                    self.features[f'{col}_return'].rolling(window).std()
                )
                self.feature_descriptions[f'{col}_vol_{window}d'] = f'{window}-day rolling volatility of {col} returns'
                
                # Volatility of levels
                self.features[f'{col}_level_vol_{window}d'] = (
                    self.data[col].rolling(window).std()
                )
                self.feature_descriptions[f'{col}_level_vol_{window}d'] = f'{window}-day rolling volatility of {col} levels'
            
            # Momentum indicators
            for window in [5, 10, 20]:
                self.features[f'{col}_momentum_{window}d'] = (
                    self.data[col] / self.data[col].shift(window) - 1
                )
                self.feature_descriptions[f'{col}_momentum_{window}d'] = f'{window}-day momentum for {col}'
                
                # Momentum strength (absolute value)
                self.features[f'{col}_momentum_strength_{window}d'] = (
                    abs(self.features[f'{col}_momentum_{window}d'])
                )
                self.feature_descriptions[f'{col}_momentum_strength_{window}d'] = f'{window}-day momentum strength for {col}'
            
            # Moving averages and deviations
            for window in [5, 10, 20, 50]:
                # Simple moving average
                self.features[f'{col}_sma_{window}d'] = self.data[col].rolling(window).mean()
                self.feature_descriptions[f'{col}_sma_{window}d'] = f'{window}-day simple moving average for {col}'
                
                # Deviation from moving average
                self.features[f'{col}_sma_dev_{window}d'] = (
                    (self.data[col] - self.features[f'{col}_sma_{window}d']) / 
                    self.features[f'{col}_sma_{window}d']
                )
                self.feature_descriptions[f'{col}_sma_dev_{window}d'] = f'Deviation from {window}-day SMA for {col}'
        
        print(f"Created {len([k for k in self.features.columns if 'return' in k or 'vol' in k or 'momentum' in k])} return-based features")
    
    def create_yield_curve_features(self):
        """
        Yield curve specific features
        """
        print("Creating yield curve features...")
        
        if 'treasury_10y' not in self.data.columns or 'treasury_13w' not in self.data.columns:
            print("Warning: Missing treasury data for yield curve analysis")
            return
        
        # Basic slope (10Y - 13W)
        self.features['yield_slope'] = (
            self.data['treasury_10y'] - self.data['treasury_13w']
        )
        self.feature_descriptions['yield_slope'] = '10Y-13W Treasury yield spread (slope)'
        
        # Yield curve level (average)
        self.features['yield_level'] = (
            (self.data['treasury_10y'] + self.data['treasury_13w']) / 2
        )
        self.feature_descriptions['yield_level'] = 'Average of 10Y and 13W Treasury yields (level)'
        
        # Slope percentiles for regime identification
        for window in [20, 60, 120]:
            self.features[f'yield_slope_percentile_{window}d'] = (
                self.features['yield_slope'].rolling(window).rank(pct=True)
            )
            self.feature_descriptions[f'yield_slope_percentile_{window}d'] = f'{window}-day percentile rank of yield slope'
        
        # Slope dynamics
        self.features['yield_slope_change'] = self.features['yield_slope'].diff()
        self.feature_descriptions['yield_slope_change'] = 'Daily change in yield slope'
        
        self.features['yield_slope_momentum_5d'] = (
            self.features['yield_slope'] / self.features['yield_slope'].shift(5) - 1
        )
        self.feature_descriptions['yield_slope_momentum_5d'] = '5-day momentum of yield slope'
        
        # Steepening/flattening indicators
        self.features['curve_steepening'] = (
            (self.features['yield_slope_change'] > 0).astype(int)
        )
        self.feature_descriptions['curve_steepening'] = 'Binary indicator for curve steepening (1=steepening, 0=flattening)'
        
        # Yield curve volatility
        for window in [10, 20]:
            self.features[f'yield_slope_vol_{window}d'] = (
                self.features['yield_slope'].rolling(window).std()
            )
            self.feature_descriptions[f'yield_slope_vol_{window}d'] = f'{window}-day volatility of yield slope'
        
        # Yield curve regime indicators
        # Flat curve indicator (slope close to zero)
        self.features['curve_flat_indicator'] = (
            abs(self.features['yield_slope']) < self.features['yield_slope'].rolling(60).std()
        ).astype(int)
        self.feature_descriptions['curve_flat_indicator'] = 'Indicator for flat yield curve periods'
        
        # Inverted curve indicator
        self.features['curve_inverted'] = (self.features['yield_slope'] < 0).astype(int)
        self.feature_descriptions['curve_inverted'] = 'Binary indicator for inverted yield curve'
        
        print(f"Created {len([k for k in self.features.columns if 'yield' in k or 'curve' in k])} yield curve features")
    
    def create_vix_features(self):
        """
        VIX-specific features
        """
        print("Creating VIX-specific features...")
        
        if 'vix' not in self.data.columns:
            print("Warning: No VIX data available")
            return
        
        # VIX level categories
        self.features['vix_low'] = (self.data['vix'] < 15).astype(int)
        self.features['vix_medium'] = ((self.data['vix'] >= 15) & (self.data['vix'] <= 25)).astype(int)
        self.features['vix_high'] = ((self.data['vix'] > 25) & (self.data['vix'] <= 40)).astype(int)
        self.features['vix_extreme'] = (self.data['vix'] > 40).astype(int)
        
        self.feature_descriptions.update({
            'vix_low': 'VIX below 15 (low volatility regime)',
            'vix_medium': 'VIX between 15-25 (medium volatility)',
            'vix_high': 'VIX between 25-40 (high volatility)',
            'vix_extreme': 'VIX above 40 (extreme volatility)'
        })
        
        # VIX percentile rankings
        for window in [20, 60, 120]:
            self.features[f'vix_percentile_{window}d'] = (
                self.data['vix'].rolling(window).rank(pct=True)
            )
            self.feature_descriptions[f'vix_percentile_{window}d'] = f'{window}-day percentile rank of VIX'
        
        # VIX mean reversion indicators
        for window in [20, 50]:
            vix_mean = self.data['vix'].rolling(window).mean()
            self.features[f'vix_mean_reversion_{window}d'] = (
                (self.data['vix'] - vix_mean) / vix_mean
            )
            self.feature_descriptions[f'vix_mean_reversion_{window}d'] = f'VIX deviation from {window}-day mean (mean reversion)'
            
            # Distance from mean in standard deviations
            vix_std = self.data['vix'].rolling(window).std()
            self.features[f'vix_zscore_{window}d'] = (
                (self.data['vix'] - vix_mean) / vix_std
            )
            self.feature_descriptions[f'vix_zscore_{window}d'] = f'VIX Z-score relative to {window}-day history'
        
        # VIX momentum and acceleration
        for window in [5, 10]:
            # Momentum
            self.features[f'vix_momentum_{window}d'] = (
                self.data['vix'] / self.data['vix'].shift(window) - 1
            )
            self.feature_descriptions[f'vix_momentum_{window}d'] = f'{window}-day VIX momentum'
            
            # Acceleration (change in momentum)
            if f'vix_momentum_{window}d' in self.features.columns:
                self.features[f'vix_acceleration_{window}d'] = (
                    self.features[f'vix_momentum_{window}d'].diff()
                )
                self.feature_descriptions[f'vix_acceleration_{window}d'] = f'{window}-day VIX acceleration'
        
        # VIX regime transition indicators
        vix_ma_short = self.data['vix'].rolling(5).mean()
        vix_ma_long = self.data['vix'].rolling(20).mean()
        
        self.features['vix_regime_shift'] = (
            (vix_ma_short > vix_ma_long).astype(int)
        )
        self.feature_descriptions['vix_regime_shift'] = 'VIX regime indicator (5-day MA above 20-day MA)'
        
        # 6. VIX spike detection
        vix_rolling_max = self.data['vix'].rolling(20).max()
        self.features['vix_spike'] = (
            self.data['vix'] / vix_rolling_max > 0.9
        ).astype(int)
        self.feature_descriptions['vix_spike'] = 'VIX spike indicator (near 20-day high)'
        
        print(f"Created {len([k for k in self.features.columns if 'vix' in k])} VIX-specific features")
    
    def create_cross_asset_features(self):
        """
        Features involving multiple variables
        """
        print("Creating cross-asset features...")
        
        available_cols = [col for col in ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread'] 
                         if col in self.data.columns]
        
        if len(available_cols) < 2:
            print("Warning: Need at least 2 variables for cross-asset features")
            return
        
        # Rolling correlations between variables
        correlation_pairs = [
            ('vix', 'treasury_10y', 'VIX vs 10Y Treasury correlation'),
            ('vix', 'credit_spread', 'VIX vs Credit Spread correlation'),
            ('treasury_10y', 'credit_spread', '10Y Treasury vs Credit Spread correlation'),
            ('yield_slope', 'vix', 'Yield Slope vs VIX correlation')
        ]
        
        for col1, col2, description in correlation_pairs:
            # Check if both columns exist (col1 might be a feature we created)
            data1 = self.data[col1] if col1 in self.data.columns else self.features.get(col1)
            data2 = self.data[col2] if col2 in self.data.columns else self.features.get(col2)
            
            if data1 is not None and data2 is not None:
                for window in [10, 20, 60]:
                    feature_name = f'{col1}_{col2}_corr_{window}d'
                    self.features[feature_name] = data1.rolling(window).corr(data2)
                    self.feature_descriptions[feature_name] = f'{window}-day {description}'
        
        # Cross-asset momentum divergence
        if 'vix' in self.data.columns and 'credit_spread' in self.data.columns:
            vix_momentum = self.data['vix'].pct_change(5)
            credit_momentum = self.data['credit_spread'].pct_change(5)
            
            self.features['vix_credit_momentum_divergence'] = (
                abs(vix_momentum - credit_momentum)
            )
            self.feature_descriptions['vix_credit_momentum_divergence'] = 'Divergence between VIX and Credit Spread 5-day momentum'
            
            # Direction agreement
            self.features['vix_credit_direction_agreement'] = (
                (np.sign(vix_momentum) == np.sign(credit_momentum)).astype(int)
            )
            self.feature_descriptions['vix_credit_direction_agreement'] = 'Agreement in direction between VIX and Credit Spread momentum'
        
        # Risk-on/Risk-off indicators
        if 'vix' in self.data.columns and 'treasury_10y' in self.data.columns:
            # Typically, risk-off periods show high VIX and falling yields
            vix_high = self.data['vix'] > self.data['vix'].rolling(20).median()
            yield_falling = self.data['treasury_10y'].diff() < 0
            
            self.features['risk_off_indicator'] = (vix_high & yield_falling).astype(int)
            self.feature_descriptions['risk_off_indicator'] = 'Risk-off market indicator (high VIX + falling yields)'
            
            self.features['risk_on_indicator'] = ((~vix_high) & (~yield_falling)).astype(int)
            self.feature_descriptions['risk_on_indicator'] = 'Risk-on market indicator (low VIX + rising yields)'
        
        # Cross-asset volatility clustering
        volatility_cols = [col for col in available_cols if col in self.data.columns]
        if len(volatility_cols) >= 2:
            # Calculate average volatility across assets
            vol_features = []
            for col in volatility_cols:
                vol_feature = self.data[col].rolling(10).std()
                vol_features.append(vol_feature)
            
            if vol_features:
                avg_volatility = pd.concat(vol_features, axis=1).mean(axis=1)
                self.features['cross_asset_volatility'] = avg_volatility
                self.feature_descriptions['cross_asset_volatility'] = 'Average 10-day volatility across all assets'
        
        # Financial stress indicator
        if len(available_cols) >= 3:
            # Normalize each variable and create composite stress indicator
            stress_components = []
            
            if 'vix' in self.data.columns:
                vix_norm = (self.data['vix'] - self.data['vix'].rolling(60).min()) / (
                    self.data['vix'].rolling(60).max() - self.data['vix'].rolling(60).min()
                )
                stress_components.append(vix_norm)
            
            if 'credit_spread' in self.data.columns:
                credit_norm = (self.data['credit_spread'] - self.data['credit_spread'].rolling(60).min()) / (
                    self.data['credit_spread'].rolling(60).max() - self.data['credit_spread'].rolling(60).min()
                )
                stress_components.append(credit_norm)
            
            if len(stress_components) >= 2:
                self.features['financial_stress_indicator'] = (
                    pd.concat(stress_components, axis=1).mean(axis=1)
                )
                self.feature_descriptions['financial_stress_indicator'] = 'Composite financial stress indicator (0-1 scale)'
        
        print(f"Created {len([k for k in self.features.columns if 'corr' in k or 'divergence' in k or 'indicator' in k])} cross-asset features")
    
    def get_feature_summary(self):
        """
        Get summary of all created features
        """
        feature_summary = {
            'total_features': len(self.features.columns),
            'feature_categories': {
                'returns': len([c for c in self.features.columns if 'return' in c or 'momentum' in c]),
                'volatility': len([c for c in self.features.columns if 'vol' in c]),
                'yield_curve': len([c for c in self.features.columns if 'yield' in c or 'curve' in c]),
                'vix': len([c for c in self.features.columns if 'vix' in c]),
                'cross_asset': len([c for c in self.features.columns if 'corr' in c or 'divergence' in c]),
                'technical': len([c for c in self.features.columns if 'sma' in c or 'percentile' in c])
            },
            'missing_values': self.features.isnull().sum().sum(),
            'feature_descriptions': self.feature_descriptions
        }
        
        return feature_summary
    
    def run_all_feature_engineering(self):
        """
        Run all feature engineering steps
        """
        print("=== RUNNING BASIC FEATURE ENGINEERING ===")
        
        self.create_basic_returns()
        self.create_yield_curve_features()
        self.create_vix_features()
        self.create_cross_asset_features()
        
        summary = self.get_feature_summary()
        
        print(f"\n=== FEATURE ENGINEERING SUMMARY ===")
        print(f"Total features created: {summary['total_features']}")
        print(f"Feature breakdown:")
        for category, count in summary['feature_categories'].items():
            print(f"  {category}: {count} features")
        print(f"Total missing values: {summary['missing_values']}")
        
        return self.features, self.feature_descriptions

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from .feature_engineering import BasicFeatureEngineer 

class AdvancedFeatureEngineer(BasicFeatureEngineer):
    def __init__(self, data, regime_labels=None):
        super().__init__(data)
        self.regime_labels = regime_labels
        self.advanced_features = pd.DataFrame(index=data.index)
        self.advanced_descriptions = {}
    
    def create_regime_dependent_features(self):
        """
        Features that behave differently by regime
        """
        if self.regime_labels is None:
            print("Warning: No regime labels provided for regime-dependent features")
            return
        
        print("Creating regime-dependent features...")
        
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        # Rolling correlations by regime
        correlation_pairs = [
            ('vix', 'credit_spread'),
            ('vix', 'treasury_10y'),
            ('treasury_10y', 'credit_spread')
        ]
        
        for col1, col2 in correlation_pairs:
            if col1 in self.data.columns and col2 in self.data.columns:
                # Calculate correlations separately for each regime
                for regime in [0, 1, 2]:  # Normal, Transition, Crisis
                    regime_mask = self.regime_labels == regime
                    regime_data = self.data[regime_mask]
                    
                    if len(regime_data) > 20:  # Minimum data points for correlation
                        # Calculate rolling correlations within regime
                        for window in [10, 20]:
                            if len(regime_data) >= window:
                                regime_corr = regime_data[col1].rolling(window, min_periods=window//2).corr(
                                    regime_data[col2]
                                )
                                
                                # Map back to full index
                                feature_name = f'{col1}_{col2}_corr_{window}d_regime{regime}'
                                self.advanced_features[feature_name] = np.nan
                                self.advanced_features.loc[regime_mask, feature_name] = regime_corr
                                
                                self.advanced_descriptions[feature_name] = (
                                    f'{window}-day correlation between {col1} and {col2} in regime {regime}'
                                )
        
        # Volatility measures by regime
        for col in available_cols:
            for regime in [0, 1, 2]:
                regime_mask = self.regime_labels == regime
                regime_data = self.data[regime_mask]
                
                if len(regime_data) > 10:
                    # Regime-specific volatility
                    for window in [5, 10, 20]:
                        if len(regime_data) >= window:
                            regime_vol = regime_data[col].rolling(window, min_periods=window//2).std()
                            
                            feature_name = f'{col}_vol_{window}d_regime{regime}'
                            self.advanced_features[feature_name] = np.nan
                            self.advanced_features.loc[regime_mask, feature_name] = regime_vol
                            
                            self.advanced_descriptions[feature_name] = (
                                f'{window}-day volatility of {col} in regime {regime}'
                            )
        
        # Mean reversion speeds by regime
        for col in available_cols:
            for regime in [0, 1, 2]:
                regime_mask = self.regime_labels == regime
                regime_data = self.data[regime_mask]
                
                if len(regime_data) > 20:
                    # Calculate mean reversion speed
                    for window in [10, 20]:
                        if len(regime_data) >= window:
                            regime_mean = regime_data[col].rolling(window, min_periods=window//2).mean()
                            regime_deviation = (regime_data[col] - regime_mean) / regime_mean
                            
                            feature_name = f'{col}_mean_reversion_regime{regime}_{window}d'
                            self.advanced_features[feature_name] = np.nan
                            self.advanced_features.loc[regime_mask, feature_name] = regime_deviation
                            
                            self.advanced_descriptions[feature_name] = (
                                f'Mean reversion indicator for {col} in regime {regime} ({window}-day)'
                            )
        
        # Cross-regime features
        # How different is current period from each regime's average?
        for col in available_cols:
            for regime in [0, 1, 2]:
                regime_mask = self.regime_labels == regime
                if regime_mask.sum() > 10:
                    regime_mean = self.data.loc[regime_mask, col].mean()
                    regime_std = self.data.loc[regime_mask, col].std()
                    
                    # Distance from regime mean
                    feature_name = f'{col}_distance_from_regime{regime}_mean'
                    self.advanced_features[feature_name] = abs(self.data[col] - regime_mean)
                    self.advanced_descriptions[feature_name] = (
                        f'Absolute distance from regime {regime} mean for {col}'
                    )
                    
                    # Z-score relative to regime
                    if regime_std > 0:
                        feature_name = f'{col}_zscore_vs_regime{regime}'
                        self.advanced_features[feature_name] = (self.data[col] - regime_mean) / regime_std
                        self.advanced_descriptions[feature_name] = (
                            f'Z-score relative to regime {regime} for {col}'
                        )
        
        print(f"Created {len([k for k in self.advanced_features.columns if 'regime' in k])} regime-dependent features")
    
    def create_momentum_features(self):
        """
        Multi-timeframe momentum indicators
        """
        print("Creating advanced momentum features...")
        
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        timeframes = [1, 3, 5, 10, 20]
        
        for tf in timeframes:
            for col in available_cols:
                # Price momentum
                momentum = self.data[col] / self.data[col].shift(tf) - 1
                feature_name = f'{col}_momentum_{tf}d'
                self.advanced_features[feature_name] = momentum
                self.advanced_descriptions[feature_name] = f'{tf}-day momentum for {col}'
                
                # Momentum strength (absolute value)
                feature_name = f'{col}_momentum_strength_{tf}d'
                self.advanced_features[feature_name] = abs(momentum)
                self.advanced_descriptions[feature_name] = f'{tf}-day momentum strength for {col}'
                
                # Momentum acceleration (change in momentum)
                if tf <= 10:  # Only for shorter timeframes
                    momentum_accel = momentum.diff()
                    feature_name = f'{col}_momentum_accel_{tf}d'
                    self.advanced_features[feature_name] = momentum_accel
                    self.advanced_descriptions[feature_name] = f'{tf}-day momentum acceleration for {col}'
                
                # Momentum persistence
                momentum_sign = np.sign(momentum)
                momentum_persistence = momentum_sign.rolling(5, min_periods=3).mean()
                feature_name = f'{col}_momentum_persistence_{tf}d'
                self.advanced_features[feature_name] = momentum_persistence
                self.advanced_descriptions[feature_name] = f'{tf}-day momentum persistence for {col}'
        
        # Cross-asset momentum divergence
        if len(available_cols) >= 2:
            for i, col1 in enumerate(available_cols):
                for col2 in available_cols[i+1:]:
                    for tf in [5, 10]:
                        mom1 = self.data[col1].pct_change(tf)
                        mom2 = self.data[col2].pct_change(tf)
                        
                        # Momentum divergence
                        feature_name = f'{col1}_{col2}_momentum_divergence_{tf}d'
                        self.advanced_features[feature_name] = abs(mom1 - mom2)
                        self.advanced_descriptions[feature_name] = (
                            f'{tf}-day momentum divergence between {col1} and {col2}'
                        )
                        
                        # Direction agreement
                        feature_name = f'{col1}_{col2}_direction_agreement_{tf}d'
                        self.advanced_features[feature_name] = (
                            (np.sign(mom1) == np.sign(mom2)).astype(int)
                        )
                        self.advanced_descriptions[feature_name] = (
                            f'{tf}-day direction agreement between {col1} and {col2}'
                        )
        
        # Momentum regime consistency
        if self.regime_labels is not None:
            for col in available_cols:
                for tf in [5, 10]:
                    momentum = self.data[col].pct_change(tf)
                    
                    # Check if momentum is consistent with current regime
                    # Crisis: expect high momentum magnitude
                    # Normal: expect low momentum magnitude
                    momentum_magnitude = abs(momentum)
                    
                    crisis_mask = self.regime_labels == 2
                    normal_mask = self.regime_labels == 0
                    
                    regime_momentum_consistency = pd.Series(0.5, index=self.data.index)  # Default neutral
                    
                    if crisis_mask.any():
                        crisis_threshold = momentum_magnitude[crisis_mask].median()
                        regime_momentum_consistency[crisis_mask] = (
                            momentum_magnitude[crisis_mask] / crisis_threshold
                        ).clip(0, 1)
                    
                    if normal_mask.any():
                        normal_threshold = momentum_magnitude[normal_mask].median()
                        regime_momentum_consistency[normal_mask] = 1 - (
                            momentum_magnitude[normal_mask] / normal_threshold
                        ).clip(0, 1)
                    
                    feature_name = f'{col}_momentum_regime_consistency_{tf}d'
                    self.advanced_features[feature_name] = regime_momentum_consistency
                    self.advanced_descriptions[feature_name] = (
                        f'{tf}-day momentum consistency with current regime for {col}'
                    )
        
        print(f"Created {len([k for k in self.advanced_features.columns if 'momentum' in k])} advanced momentum features")
    
    def create_mean_reversion_features(self):
        """
        Mean reversion indicators
        """
        print("Creating mean reversion features...")
        
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        for col in available_cols:
            # Distance from multiple moving averages
            for window in [5, 10, 20, 50, 100]:
                if len(self.data) >= window:
                    ma = self.data[col].rolling(window, min_periods=window//2).mean()
                    
                    # Relative distance
                    feature_name = f'{col}_ma_deviation_{window}d'
                    self.advanced_features[feature_name] = (self.data[col] - ma) / ma
                    self.advanced_descriptions[feature_name] = (
                        f'Relative deviation from {window}-day MA for {col}'
                    )
                    
                    # Absolute distance
                    feature_name = f'{col}_ma_abs_deviation_{window}d'
                    self.advanced_features[feature_name] = abs(self.data[col] - ma)
                    self.advanced_descriptions[feature_name] = (
                        f'Absolute deviation from {window}-day MA for {col}'
                    )
            
            # Z-scores relative to historical periods
            for window in [20, 60, 120]:
                if len(self.data) >= window:
                    rolling_mean = self.data[col].rolling(window, min_periods=window//2).mean()
                    rolling_std = self.data[col].rolling(window, min_periods=window//2).std()
                    
                    feature_name = f'{col}_zscore_{window}d'
                    self.advanced_features[feature_name] = (
                        (self.data[col] - rolling_mean) / rolling_std
                    )
                    self.advanced_descriptions[feature_name] = (
                        f'Z-score relative to {window}-day history for {col}'
                    )
            
            # Mean reversion probability indicators
            for window in [10, 20]:
                if len(self.data) >= window * 2:
                    # Calculate how often price reverts to mean within X days
                    ma = self.data[col].rolling(window, min_periods=window//2).mean()
                    deviation = self.data[col] - ma
                    
                    # Look ahead to see if price reverts (for historical analysis)
                    reversion_periods = [3, 5, 10]
                    for rev_period in reversion_periods:
                        will_revert = pd.Series(False, index=self.data.index)
                        
                        for i in range(len(self.data) - rev_period):
                            current_dev = deviation.iloc[i]
                            if abs(current_dev) > 0:  # Only if there's deviation
                                future_devs = deviation.iloc[i+1:i+rev_period+1]
                                # Check if sign changes (crosses mean)
                                sign_changes = (np.sign(future_devs) != np.sign(current_dev)).any()
                                will_revert.iloc[i] = sign_changes
                        
                        feature_name = f'{col}_mean_reversion_prob_{window}d_{rev_period}d'
                        self.advanced_features[feature_name] = will_revert.rolling(
                            window, min_periods=window//2
                        ).mean()
                        self.advanced_descriptions[feature_name] = (
                            f'Historical probability of mean reversion within {rev_period} days for {col}'
                        )
            
            # Mean reversion strength
            for window in [20, 50]:
                if len(self.data) >= window:
                    ma = self.data[col].rolling(window, min_periods=window//2).mean()
                    deviation = abs(self.data[col] - ma)
                    max_historical_dev = deviation.rolling(window*2, min_periods=window).max()
                    
                    feature_name = f'{col}_mean_reversion_strength_{window}d'
                    self.advanced_features[feature_name] = deviation / max_historical_dev
                    self.advanced_descriptions[feature_name] = (
                        f'Mean reversion strength for {col} (current dev / max historical dev)'
                    )
        
        print(f"Created {len([k for k in self.advanced_features.columns if 'reversion' in k or 'zscore' in k])} mean reversion features")
    
    def create_structural_features(self):
        """
        Features indicating structural changes
        """
        print("Creating structural change features...")
        
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        for col in available_cols:
            # Rolling correlation changes
            if len(available_cols) > 1:
                other_cols = [c for c in available_cols if c != col]
                for other_col in other_cols:
                    for window in [20, 60]:
                        if len(self.data) >= window * 2:
                            # Current correlation
                            current_corr = self.data[col].rolling(window, min_periods=window//2).corr(
                                self.data[other_col]
                            )
                            
                            # Historical correlation
                            hist_corr = current_corr.shift(window)
                            
                            # Correlation change
                            feature_name = f'{col}_{other_col}_corr_change_{window}d'
                            self.advanced_features[feature_name] = abs(current_corr - hist_corr)
                            self.advanced_descriptions[feature_name] = (
                                f'Change in {window}-day correlation between {col} and {other_col}'
                            )
            
            # Volatility regime changes
            for window in [10, 20, 30]:
                if len(self.data) >= window * 3:
                    current_vol = self.data[col].rolling(window, min_periods=window//2).std()
                    historical_vol = current_vol.shift(window)
                    
                    # Volatility ratio
                    feature_name = f'{col}_vol_regime_change_{window}d'
                    self.advanced_features[feature_name] = current_vol / historical_vol
                    self.advanced_descriptions[feature_name] = (
                        f'Volatility regime change indicator for {col} ({window}-day)'
                    )
                    
                    # Volatility breakout indicator
                    vol_threshold = current_vol.rolling(window*2, min_periods=window).quantile(0.8)
                    feature_name = f'{col}_vol_breakout_{window}d'
                    self.advanced_features[feature_name] = (current_vol > vol_threshold).astype(int)
                    self.advanced_descriptions[feature_name] = (
                        f'Volatility breakout indicator for {col} ({window}-day)'
                    )
            
            # Trend change indicators
            for window in [5, 10, 20]:
                if len(self.data) >= window * 2:
                    # Short-term vs long-term trend
                    short_ma = self.data[col].rolling(window, min_periods=window//2).mean()
                    long_ma = self.data[col].rolling(window*2, min_periods=window).mean()
                    
                    # Trend direction
                    trend_direction = (short_ma > long_ma).astype(int)
                    trend_change = (trend_direction != trend_direction.shift(1)).astype(int)
                    
                    feature_name = f'{col}_trend_change_{window}d'
                    self.advanced_features[feature_name] = trend_change
                    self.advanced_descriptions[feature_name] = (
                        f'Trend change indicator for {col} ({window}-day)'
                    )
                    
                    # Trend strength
                    feature_name = f'{col}_trend_strength_{window}d'
                    self.advanced_features[feature_name] = abs(short_ma - long_ma) / long_ma
                    self.advanced_descriptions[feature_name] = (
                        f'Trend strength for {col} ({window}-day)'
                    )
            
            # Structural break indicators using statistics
            for window in [30, 60]:
                if len(self.data) >= window * 2:
                    # Rolling mean stability
                    rolling_mean = self.data[col].rolling(window, min_periods=window//2).mean()
                    mean_stability = rolling_mean.rolling(window, min_periods=window//2).std()
                    
                    feature_name = f'{col}_mean_stability_{window}d'
                    self.advanced_features[feature_name] = mean_stability
                    self.advanced_descriptions[feature_name] = (
                        f'Mean stability indicator for {col} ({window}-day)'
                    )
                    
                    # Distribution change indicator
                    current_skew = self.data[col].rolling(window, min_periods=window//2).skew()
                    current_kurt = self.data[col].rolling(window, min_periods=window//2).kurt()
                    
                    feature_name = f'{col}_skewness_{window}d'
                    self.advanced_features[feature_name] = current_skew
                    self.advanced_descriptions[feature_name] = (
                        f'Rolling skewness for {col} ({window}-day)'
                    )
                    
                    feature_name = f'{col}_kurtosis_{window}d'
                    self.advanced_features[feature_name] = current_kurt
                    self.advanced_descriptions[feature_name] = (
                        f'Rolling kurtosis for {col} ({window}-day)'
                    )
        
        print(f"Created {len([k for k in self.advanced_features.columns if any(x in k for x in ['change', 'breakout', 'stability'])])} structural change features")
    
    def create_interaction_features(self):
        """
        Create interaction features between variables
        """
        print("Creating interaction features...")
        
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        # Multiplicative interactions
        for i, col1 in enumerate(available_cols):
            for col2 in available_cols[i+1:]:
                # Simple multiplication
                feature_name = f'{col1}_{col2}_interaction'
                self.advanced_features[feature_name] = self.data[col1] * self.data[col2]
                self.advanced_descriptions[feature_name] = f'Multiplicative interaction between {col1} and {col2}'
                
                # Normalized interaction
                col1_norm = (self.data[col1] - self.data[col1].mean()) / self.data[col1].std()
                col2_norm = (self.data[col2] - self.data[col2].mean()) / self.data[col2].std()
                
                feature_name = f'{col1}_{col2}_interaction_norm'
                self.advanced_features[feature_name] = col1_norm * col2_norm
                self.advanced_descriptions[feature_name] = f'Normalized multiplicative interaction between {col1} and {col2}'
        
        # Ratio features
        ratio_pairs = [
            ('vix', 'treasury_10y', 'Risk premium ratio'),
            ('credit_spread', 'treasury_10y', 'Credit risk ratio'),
            ('treasury_10y', 'treasury_13w', 'Yield curve slope ratio')
        ]
        
        for col1, col2, description in ratio_pairs:
            if col1 in self.data.columns and col2 in self.data.columns:
                # Avoid division by zero
                denominator = self.data[col2].replace(0, np.nan)
                
                feature_name = f'{col1}_{col2}_ratio'
                self.advanced_features[feature_name] = self.data[col1] / denominator
                self.advanced_descriptions[feature_name] = f'{description} ({col1}/{col2})'
        
        print(f"Created {len([k for k in self.advanced_features.columns if 'interaction' in k or 'ratio' in k])} interaction features")
    
    def run_all_advanced_features(self):
        """
        Run all advanced feature engineering steps
        """
        print("\n=== RUNNING ADVANCED FEATURE ENGINEERING ===")
        
        # Create all advanced features
        self.create_regime_dependent_features()
        self.create_momentum_features()
        self.create_mean_reversion_features()
        self.create_structural_features()
        self.create_interaction_features()
        
        # Combine with basic features
        all_features = pd.concat([self.features, self.advanced_features], axis=1)
        all_descriptions = {**self.feature_descriptions, **self.advanced_descriptions}
        
        # Summary
        print(f"\n=== ADVANCED FEATURE ENGINEERING SUMMARY ===")
        print(f"Basic features: {len(self.features.columns)}")
        print(f"Advanced features: {len(self.advanced_features.columns)}")
        print(f"Total features: {len(all_features.columns)}")
        
        feature_categories = {
            'regime_dependent': len([k for k in all_features.columns if 'regime' in k]),
            'momentum': len([k for k in all_features.columns if 'momentum' in k]),
            'mean_reversion': len([k for k in all_features.columns if 'reversion' in k or 'zscore' in k]),
            'structural': len([k for k in all_features.columns if any(x in k for x in ['change', 'breakout', 'stability'])]),
            'interaction': len([k for k in all_features.columns if 'interaction' in k or 'ratio' in k]),
            'basic': len([k for k in all_features.columns if not any(x in k for x in ['regime', 'momentum', 'reversion', 'zscore', 'change', 'breakout', 'stability', 'interaction', 'ratio'])])
        }
        
        print("Feature categories:")
        for category, count in feature_categories.items():
            print(f"  {category}: {count}")
        
        print(f"Total missing values: {all_features.isnull().sum().sum()}")
        
        return all_features, all_descriptions