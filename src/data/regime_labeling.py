# src/data/regime_labeling.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ManualRegimeLabeler:
    def __init__(self, data):
        self.data = data.copy()
        self.regime_labels = None
        self.regime_continuous = None
        self.regime_history = {}
        
        # Validate required columns
        if 'date' not in self.data.columns:
            raise ValueError("Data must contain 'date' column")
        if 'vix' not in self.data.columns:
            raise ValueError("Data must contain 'vix' column for regime identification")
    
    def identify_crisis_periods(self):
        """
        Manually identify crisis periods based on VIX and market knowledge
        """
        print("Identifying crisis periods...")
        
        crisis_conditions = pd.Series(False, index=self.data.index)
        crisis_events = []
        
        # VIX-based crisis identification
        high_vix = self.data['vix'] > 30
        sustained_high_vix = high_vix.rolling(3, min_periods=1).sum() >= 2  # 2 out of 3 days
        
        # Credit spread-based crisis identification (if available)
        if 'credit_spread' in self.data.columns:
            credit_percentile = self.data['credit_spread'].rolling(60, min_periods=20).rank(pct=True)
            high_credit = credit_percentile > 0.9
            crisis_conditions = crisis_conditions | high_credit
            crisis_events.append("High credit spreads (>90th percentile)")
        
        # Yield volatility-based identification
        if 'treasury_10y' in self.data.columns:
            yield_vol = self.data['treasury_10y'].rolling(5, min_periods=3).std()
            yield_vol_threshold = yield_vol.rolling(60, min_periods=20).quantile(0.9)
            high_yield_vol = yield_vol > yield_vol_threshold
            crisis_conditions = crisis_conditions | high_yield_vol
            crisis_events.append("High yield volatility (>90th percentile)")
        
        # Extreme VIX spikes (VIX > 40)
        extreme_vix = self.data['vix'] > 40
        crisis_conditions = crisis_conditions | extreme_vix
        crisis_events.append("Extreme VIX (>40)")
        
        # Sustained high VIX periods
        crisis_conditions = crisis_conditions | sustained_high_vix
        crisis_events.append("Sustained high VIX (>30)")
        
        # Known historical crisis periods 
        historical_crises = [
            ('2018-02-01', '2018-03-31', 'Volatility Spike'),
            ('2018-10-01', '2018-12-31', 'Q4 2018 Selloff'),
            ('2020-02-01', '2020-05-31', 'COVID-19 Crisis'),
            ('2022-02-01', '2022-04-30', 'Russia-Ukraine War'),
            ('2023-03-01', '2023-03-31', 'Banking Crisis (SVB)')
        ]
        
        data_start = self.data['date'].min()
        data_end = self.data['date'].max()
        
        for start_str, end_str, description in historical_crises:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            
            # Check if crisis period overlaps with our data
            if start_date <= data_end and end_date >= data_start:
                crisis_mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
                if crisis_mask.any():
                    crisis_conditions = crisis_conditions | crisis_mask
                    crisis_events.append(f"Historical event: {description}")
                    print(f"  Identified historical crisis: {description} ({start_str} to {end_str})")
        
        self.regime_history['crisis_periods'] = {
            'conditions': crisis_conditions,
            'events': crisis_events,
            'count': crisis_conditions.sum(),
            'percentage': (crisis_conditions.sum() / len(self.data)) * 100
        }
        
        print(f"  Identified {crisis_conditions.sum()} crisis days ({(crisis_conditions.sum()/len(self.data)*100):.1f}%)")
        return crisis_conditions
        
    def identify_normal_periods(self):
        """
        Identify stable, low-volatility periods
        """
        print("Identifying normal periods...")
        
        normal_conditions = pd.Series(True, index=self.data.index)  # Start with all days as normal
        normal_criteria = []
        
        # Low VIX periods (VIX < 20 for extended periods)
        low_vix = self.data['vix'] < 20
        sustained_low_vix = low_vix.rolling(5, min_periods=3).sum() >= 4  # 4 out of 5 days
        
        # Stable yield curves (if available)
        if 'treasury_10y' in self.data.columns and 'treasury_13w' in self.data.columns:
            yield_slope = self.data['treasury_10y'] - self.data['treasury_13w']
            yield_slope_stable = abs(yield_slope.diff()) < yield_slope.rolling(30, min_periods=10).std()
            normal_conditions = normal_conditions & yield_slope_stable
            normal_criteria.append("Stable yield curve")
        
        # Tight credit spreads (if available)
        if 'credit_spread' in self.data.columns:
            credit_percentile = self.data['credit_spread'].rolling(60, min_periods=20).rank(pct=True)
            tight_credit = credit_percentile < 0.5  # Below median
            normal_conditions = normal_conditions & tight_credit
            normal_criteria.append("Tight credit spreads (<50th percentile)")
        
        # Low cross-asset volatility
        available_assets = [col for col in ['vix', 'treasury_10y', 'credit_spread'] if col in self.data.columns]
        if len(available_assets) >= 2:
            # Calculate average volatility across assets
            asset_vols = []
            for asset in available_assets:
                asset_vol = self.data[asset].rolling(10, min_periods=5).std()
                asset_vols.append(asset_vol)
            
            avg_vol = pd.concat(asset_vols, axis=1).mean(axis=1)
            low_cross_vol = avg_vol < avg_vol.rolling(60, min_periods=20).quantile(0.4)
            normal_conditions = normal_conditions & low_cross_vol
            normal_criteria.append("Low cross-asset volatility")
        
        # Apply VIX condition
        normal_conditions = normal_conditions & sustained_low_vix
        normal_criteria.append("Sustained low VIX (<20)")
        
        self.regime_history['normal_periods'] = {
            'conditions': normal_conditions,
            'criteria': normal_criteria,
            'count': normal_conditions.sum(),
            'percentage': (normal_conditions.sum() / len(self.data)) * 100
        }
        
        print(f"  Identified {normal_conditions.sum()} normal days ({(normal_conditions.sum()/len(self.data)*100):.1f}%)")
        return normal_conditions
        
    def identify_transition_periods(self):
        """
        Identify periods of regime transition
        """
        print("Identifying transition periods...")
        
        transition_conditions = pd.Series(False, index=self.data.index)
        transition_indicators = []
        
        # Rapid VIX changes
        vix_change = abs(self.data['vix'].diff())
        vix_change_threshold = vix_change.rolling(30, min_periods=10).quantile(0.8)
        rapid_vix_change = vix_change > vix_change_threshold
        transition_conditions = transition_conditions | rapid_vix_change
        transition_indicators.append("Rapid VIX changes")
        
        # VIX in transition zone (20-30)
        vix_transition = (self.data['vix'] >= 20) & (self.data['vix'] <= 30)
        transition_conditions = transition_conditions | vix_transition
        transition_indicators.append("VIX in transition zone (20-30)")
        
        # Yield curve shape changes (if available)
        if 'treasury_10y' in self.data.columns and 'treasury_13w' in self.data.columns:
            yield_slope = self.data['treasury_10y'] - self.data['treasury_13w']
            slope_change = abs(yield_slope.diff())
            slope_change_threshold = slope_change.rolling(30, min_periods=10).quantile(0.8)
            rapid_slope_change = slope_change > slope_change_threshold
            transition_conditions = transition_conditions | rapid_slope_change
            transition_indicators.append("Rapid yield curve changes")
        
        # Credit spread movements (if available)
        if 'credit_spread' in self.data.columns:
            credit_change = abs(self.data['credit_spread'].pct_change())
            credit_change_threshold = credit_change.rolling(30, min_periods=10).quantile(0.8)
            rapid_credit_change = credit_change > credit_change_threshold
            transition_conditions = transition_conditions | rapid_credit_change
            transition_indicators.append("Rapid credit spread movements")
        
        # Regime crossing indicators
        # When VIX crosses key thresholds (20, 30)
        vix_cross_20 = ((self.data['vix'] > 20) != (self.data['vix'].shift(1) > 20))
        vix_cross_30 = ((self.data['vix'] > 30) != (self.data['vix'].shift(1) > 30))
        regime_crossing = vix_cross_20 | vix_cross_30
        
        # Extend crossing effect to nearby periods
        crossing_extended = regime_crossing.rolling(5, center=True, min_periods=1).max().fillna(False)
        transition_conditions = transition_conditions | crossing_extended
        transition_indicators.append("VIX regime threshold crossings")
        
        self.regime_history['transition_periods'] = {
            'conditions': transition_conditions,
            'indicators': transition_indicators,
            'count': transition_conditions.sum(),
            'percentage': (transition_conditions.sum() / len(self.data)) * 100
        }
        
        print(f"  Identified {transition_conditions.sum()} transition days ({(transition_conditions.sum()/len(self.data)*100):.1f}%)")
        return transition_conditions
    
    def create_regime_labels(self):
        """
        Create final regime classification
        0: Normal/Low Volatility
        1: Transition/Moderate Volatility  
        2: Crisis/High Volatility
        """
        print("\n=== CREATING REGIME LABELS ===")
        
        # Get regime conditions
        crisis_periods = self.identify_crisis_periods()
        normal_periods = self.identify_normal_periods()
        transition_periods = self.identify_transition_periods()
        
        # Initialize with transition as default (1)
        self.regime_labels = pd.Series(1, index=self.data.index)
        
        # Apply labels in order of priority (crisis overrides transition, normal overrides transition)
        self.regime_labels[normal_periods] = 0  # Normal
        self.regime_labels[crisis_periods] = 2   # Crisis
        
        # Clean up overlaps - crisis takes precedence over everything
        self.regime_labels[crisis_periods] = 2
        
        # Apply some smoothing to reduce noise
        self.regime_labels = self._smooth_regime_labels(self.regime_labels)
        
        # Create summary
        regime_counts = self.regime_labels.value_counts().sort_index()
        
        print(f"\nFinal regime distribution:")
        print(f"  Normal (0):     {regime_counts.get(0, 0):4d} days ({regime_counts.get(0, 0)/len(self.data)*100:5.1f}%)")
        print(f"  Transition (1): {regime_counts.get(1, 0):4d} days ({regime_counts.get(1, 0)/len(self.data)*100:5.1f}%)")
        print(f"  Crisis (2):     {regime_counts.get(2, 0):4d} days ({regime_counts.get(2, 0)/len(self.data)*100:5.1f}%)")
        
        return self.regime_labels
    
    def _smooth_regime_labels(self, labels, window=3):
        """
        Apply smoothing to reduce single-day regime switches
        """
        smoothed = labels.copy()
        
        for i in range(window, len(labels) - window):
            # Get window around current point
            window_data = labels.iloc[i-window:i+window+1]
            
            # If current point is different from majority in window, consider changing it
            current_label = labels.iloc[i]
            mode_label = window_data.mode().iloc[0] if not window_data.mode().empty else current_label
            
            # Only change if there's strong evidence (more than half the window)
            if (window_data == mode_label).sum() > (len(window_data) // 2 + 1):
                smoothed.iloc[i] = mode_label
        
        return smoothed
    
    def create_continuous_regime_score(self):
        """
        Create continuous regime score instead of discrete labels (0-1 scale)
        """
        print("Creating continuous regime score...")
        
        # Combine multiple regime indicators
        vix_score = (self.data['vix'] - self.data['vix'].rolling(60, min_periods=20).min()) / (
            self.data['vix'].rolling(60, min_periods=20).max() - 
            self.data['vix'].rolling(60, min_periods=20).min()
        )
        
        score_components = [vix_score * 0.6]  # VIX gets 60% weight
        
        if 'credit_spread' in self.data.columns:
            credit_score = (self.data['credit_spread'] - self.data['credit_spread'].rolling(60, min_periods=20).min()) / (
                self.data['credit_spread'].rolling(60, min_periods=20).max() - 
                self.data['credit_spread'].rolling(60, min_periods=20).min()
            )
            score_components.append(credit_score * 0.3)  # Credit spread gets 30% weight
        
        if 'treasury_10y' in self.data.columns:
            # For treasury yields, higher volatility indicates stress
            yield_vol = self.data['treasury_10y'].rolling(10, min_periods=5).std()
            yield_vol_score = (yield_vol - yield_vol.rolling(60, min_periods=20).min()) / (
                yield_vol.rolling(60, min_periods=20).max() - 
                yield_vol.rolling(60, min_periods=20).min()
            )
            score_components.append(yield_vol_score * 0.1)  # Yield volatility gets 10% weight
        
        # Combine components
        regime_score = pd.concat(score_components, axis=1).sum(axis=1)
        
        # Ensure 0-1 range and handle missing values
        self.regime_continuous = regime_score.clip(0, 1).fillna(0.5)
        
        print(f"Continuous regime score: mean={self.regime_continuous.mean():.3f}, std={self.regime_continuous.std():.3f}")
        
        return self.regime_continuous
    
    def validate_regime_labels(self):
        """
        Validate regime labels against known market events and VIX patterns
        """
        print("\n=== REGIME VALIDATION ===")
        
        if self.regime_labels is None:
            print("No regime labels to validate. Run create_regime_labels() first.")
            return
        
        validation_results = {}
        
        # VIX consistency check
        vix_by_regime = self.data.groupby(self.regime_labels)['vix'].agg(['mean', 'std', 'min', 'max'])
        print("\nVIX statistics by regime:")
        print(vix_by_regime.round(2))
        
        # Validate expected ordering: Normal < Transition < Crisis
        expected_order = vix_by_regime['mean'].iloc[0] < vix_by_regime['mean'].iloc[1] < vix_by_regime['mean'].iloc[2]
        validation_results['vix_ordering'] = expected_order
        print(f"VIX ordering correct (Normal < Transition < Crisis): {expected_order}")
        
        # Regime transition frequency
        regime_changes = (self.regime_labels != self.regime_labels.shift(1)).sum()
        avg_regime_duration = len(self.regime_labels) / regime_changes
        validation_results['regime_changes'] = regime_changes
        validation_results['avg_duration'] = avg_regime_duration
        print(f"Regime changes: {regime_changes}, Average duration: {avg_regime_duration:.1f} days")
        
        # Crisis period validation
        crisis_periods = self.regime_labels == 2
        if crisis_periods.any():
            crisis_vix_avg = self.data.loc[crisis_periods, 'vix'].mean()
            non_crisis_vix_avg = self.data.loc[~crisis_periods, 'vix'].mean()
            validation_results['crisis_vix_ratio'] = crisis_vix_avg / non_crisis_vix_avg
            print(f"Crisis VIX average: {crisis_vix_avg:.1f}, Non-crisis: {non_crisis_vix_avg:.1f}")
            print(f"Crisis/Non-crisis VIX ratio: {crisis_vix_avg/non_crisis_vix_avg:.2f}")
        
        return validation_results
    
    def plot_regime_analysis(self):
        """
        Create comprehensive regime analysis plots
        """
        if self.regime_labels is None:
            print("No regime labels to plot. Run create_regime_labels() first.")
            return None
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Regime Analysis Dashboard', fontsize=16)
        
        # VIX time series with regime coloring
        ax1 = axes[0, 0]
        for regime in [0, 1, 2]:
            regime_mask = self.regime_labels == regime
            regime_colors = ['green', 'orange', 'red']
            regime_names = ['Normal', 'Transition', 'Crisis']
            
            ax1.scatter(self.data.loc[regime_mask, 'date'], 
                       self.data.loc[regime_mask, 'vix'],
                       c=regime_colors[regime], alpha=0.6, s=10, 
                       label=regime_names[regime])
        
        ax1.set_title('VIX with Regime Coloring')
        ax1.set_ylabel('VIX')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regime over time
        ax2 = axes[0, 1]
        ax2.plot(self.data['date'], self.regime_labels, linewidth=2)
        ax2.set_title('Regime Evolution Over Time')
        ax2.set_ylabel('Regime')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Normal', 'Transition', 'Crisis'])
        ax2.grid(True, alpha=0.3)
        
        # VIX distribution by regime
        ax3 = axes[1, 0]
        for regime in [0, 1, 2]:
            regime_mask = self.regime_labels == regime
            regime_colors = ['green', 'orange', 'red']
            regime_names = ['Normal', 'Transition', 'Crisis']
            
            if regime_mask.any():
                ax3.hist(self.data.loc[regime_mask, 'vix'], bins=20, alpha=0.6, 
                        color=regime_colors[regime], label=regime_names[regime])
        
        ax3.set_title('VIX Distribution by Regime')
        ax3.set_xlabel('VIX')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Regime duration analysis
        ax4 = axes[1, 1]
        regime_changes = self.regime_labels != self.regime_labels.shift(1)
        regime_periods = []
        current_regime = None
        current_duration = 0
        
        for i, (regime, is_change) in enumerate(zip(self.regime_labels, regime_changes)):
            if is_change or i == 0:
                if current_regime is not None:
                    regime_periods.append((current_regime, current_duration))
                current_regime = regime
                current_duration = 1
            else:
                current_duration += 1
        
        # Add final period
        if current_regime is not None:
            regime_periods.append((current_regime, current_duration))
        
        # Plot duration distribution
        regime_durations = {0: [], 1: [], 2: []}
        for regime, duration in regime_periods:
            regime_durations[regime].append(duration)
        
        regime_colors = ['green', 'orange', 'red']
        regime_names = ['Normal', 'Transition', 'Crisis']
        
        for regime in [0, 1, 2]:
            if regime_durations[regime]:
                ax4.hist(regime_durations[regime], bins=15, alpha=0.6, 
                        color=regime_colors[regime], label=f'{regime_names[regime]} (avg: {np.mean(regime_durations[regime]):.1f})')
        
        ax4.set_title('Regime Duration Distribution')
        ax4.set_xlabel('Duration (days)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Continuous regime score (if available)
        ax5 = axes[2, 0]
        if self.regime_continuous is not None:
            ax5.plot(self.data['date'], self.regime_continuous, alpha=0.8)
            ax5.set_title('Continuous Regime Score')
            ax5.set_ylabel('Regime Score (0-1)')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No continuous score\navailable', 
                    transform=ax5.transAxes, ha='center', va='center')
        
        # Regime transition matrix
        ax6 = axes[2, 1]
        transition_matrix = np.zeros((3, 3))
        
        for i in range(1, len(self.regime_labels)):
            prev_regime = self.regime_labels.iloc[i-1]
            curr_regime = self.regime_labels.iloc[i]
            transition_matrix[prev_regime, curr_regime] += 1
        
        # Normalize by row sums
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        transition_matrix = np.nan_to_num(transition_matrix)
        
        sns.heatmap(transition_matrix, annot=True, fmt='.3f', 
                   xticklabels=['Normal', 'Transition', 'Crisis'],
                   yticklabels=['Normal', 'Transition', 'Crisis'],
                   cmap='Blues', ax=ax6)
        ax6.set_title('Regime Transition Probabilities')
        ax6.set_xlabel('To Regime')
        ax6.set_ylabel('From Regime')
        
        plt.tight_layout()
        return fig
    
    def get_regime_summary(self):
        """
        Get comprehensive summary of regime analysis
        """
        if self.regime_labels is None:
            return {"error": "No regime labels available"}
        
        summary = {
            'regime_distribution': self.regime_labels.value_counts().sort_index().to_dict(),
            'regime_percentages': (self.regime_labels.value_counts().sort_index() / len(self.regime_labels) * 100).round(2).to_dict(),
            'vix_by_regime': self.data.groupby(self.regime_labels)['vix'].agg(['mean', 'std', 'min', 'max']).round(2).to_dict(),
            'regime_changes': (self.regime_labels != self.regime_labels.shift(1)).sum(),
            'avg_regime_duration': len(self.regime_labels) / (self.regime_labels != self.regime_labels.shift(1)).sum(),
            'data_period': {
                'start': self.data['date'].min(),
                'end': self.data['date'].max(),
                'total_days': len(self.data)
            },
            'regime_history': self.regime_history
        }
        
        return summary