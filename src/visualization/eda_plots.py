# src/visualization/eda_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings

class EDAVisualizer:
    def __init__(self, data, features=None):
        self.data = data
        self.features = features if features is not None else pd.DataFrame()
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_time_series(self):
        """
        Create time series plots for all variables
        """
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        if not available_cols:
            print("No financial columns available for time series plots")
            return None
        
        n_cols = len(available_cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=(15, 4*n_cols))
        
        if n_cols == 1:
            axes = [axes]
        
        fig.suptitle('Financial Time Series Analysis', fontsize=16, y=0.98)
        
        for i, col in enumerate(available_cols):
            ax = axes[i]
            
            # Main time series
            if 'date' in self.data.columns:
                ax.plot(self.data['date'], self.data[col], linewidth=1, alpha=0.8, label=col.title())
            else:
                ax.plot(self.data.index, self.data[col], linewidth=1, alpha=0.8, label=col.title())
            
            # Add moving averages
            ma_20 = self.data[col].rolling(20).mean()
            ma_50 = self.data[col].rolling(50).mean()
            
            if 'date' in self.data.columns:
                ax.plot(self.data['date'], ma_20, '--', alpha=0.7, linewidth=1, label='20-day MA')
                ax.plot(self.data['date'], ma_50, '--', alpha=0.7, linewidth=1, label='50-day MA')
            else:
                ax.plot(self.data.index, ma_20, '--', alpha=0.7, linewidth=1, label='20-day MA')
                ax.plot(self.data.index, ma_50, '--', alpha=0.7, linewidth=1, label='50-day MA')
            
            # Highlight extreme periods
            if col == 'vix':
                extreme_periods = self.data['vix'] > 40
                if extreme_periods.any():
                    extreme_dates = self.data.loc[extreme_periods, 'date'] if 'date' in self.data.columns else self.data.index[extreme_periods]
                    extreme_values = self.data.loc[extreme_periods, col]
                    ax.scatter(extreme_dates, extreme_values, color='red', s=20, alpha=0.8, label='Extreme (>40)')
            
            ax.set_title(f'{col.title()} Over Time')
            ax.set_ylabel(col.title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if i == len(available_cols) - 1:  # Last subplot
                ax.set_xlabel('Date' if 'date' in self.data.columns else 'Index')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_analysis(self):
        """
        Correlation analysis and heatmaps
        """
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        if len(available_cols) < 2:
            print("Need at least 2 variables for correlation analysis")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Analysis Dashboard', fontsize=16)
        
        # 1. Static correlation matrix
        ax1 = axes[0, 0]
        corr_matrix = self.data[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax1, fmt='.3f')
        ax1.set_title('Static Correlation Matrix')
        
        # 2. Rolling correlation (VIX vs others)
        ax2 = axes[0, 1]
        if 'vix' in available_cols and len(available_cols) > 1:
            for col in available_cols:
                if col != 'vix':
                    rolling_corr = self.data['vix'].rolling(30).corr(self.data[col])
                    if 'date' in self.data.columns:
                        ax2.plot(self.data['date'], rolling_corr, label=f'VIX vs {col.title()}', alpha=0.7)
                    else:
                        ax2.plot(self.data.index, rolling_corr, label=f'VIX vs {col.title()}', alpha=0.7)
            
            ax2.set_title('30-Day Rolling Correlations with VIX')
            ax2.set_ylabel('Correlation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'VIX not available', transform=ax2.transAxes, ha='center')
        
        # 3. Scatter plot matrix (subset)
        ax3 = axes[1, 0]
        if len(available_cols) >= 2:
            col1, col2 = available_cols[0], available_cols[1]
            scatter = ax3.scatter(self.data[col1], self.data[col2], alpha=0.6, s=20)
            
            # Add trend line
            z = np.polyfit(self.data[col1].dropna(), self.data[col2].dropna(), 1)
            p = np.poly1d(z)
            ax3.plot(self.data[col1], p(self.data[col1]), "r--", alpha=0.8)
            
            # Calculate correlation
            corr_val = self.data[col1].corr(self.data[col2])
            ax3.set_title(f'{col1.title()} vs {col2.title()}\nCorrelation: {corr_val:.3f}')
            ax3.set_xlabel(col1.title())
            ax3.set_ylabel(col2.title())
            ax3.grid(True, alpha=0.3)
        
        # 4. Distribution of correlations over time
        ax4 = axes[1, 1]
        if len(available_cols) >= 2:
            # Calculate rolling correlations for all pairs
            window = 30
            correlations = []
            
            for i in range(len(available_cols)):
                for j in range(i+1, len(available_cols)):
                    col1, col2 = available_cols[i], available_cols[j]
                    rolling_corr = self.data[col1].rolling(window).corr(self.data[col2])
                    correlations.extend(rolling_corr.dropna().values)
            
            if correlations:
                ax4.hist(correlations, bins=30, alpha=0.7, edgecolor='black')
                ax4.set_title('Distribution of Rolling Correlations')
                ax4.set_xlabel('Correlation Value')
                ax4.set_ylabel('Frequency')
                ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_regime_identification(self):
        """
        Visual regime identification
        """
        if 'vix' not in self.data.columns:
            print("VIX data required for regime identification plots")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Market Regime Identification Analysis', fontsize=16)
        
        # 1. VIX vs Yield Slope scatter plot
        ax1 = axes[0, 0]
        if 'treasury_10y' in self.data.columns and 'treasury_13w' in self.data.columns:
            yield_slope = self.data['treasury_10y'] - self.data['treasury_13w']
            
            # Color points by VIX level
            vix_colors = ['green' if v < 20 else 'orange' if v < 30 else 'red' for v in self.data['vix']]
            
            scatter = ax1.scatter(yield_slope, self.data['vix'], c=vix_colors, alpha=0.6, s=20)
            ax1.set_xlabel('Yield Curve Slope (10Y - 13W)')
            ax1.set_ylabel('VIX')
            ax1.set_title('VIX vs Yield Curve Slope\n(Green: VIX<20, Orange: 20-30, Red: >30)')
            ax1.grid(True, alpha=0.3)
            
            # Add regime boundaries
            ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='VIX=20')
            ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='VIX=30')
            ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='Flat Curve')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Treasury data not available', transform=ax1.transAxes, ha='center')
        
        # 2. Time series with regime periods highlighted
        ax2 = axes[0, 1]
        if 'date' in self.data.columns:
            ax2.plot(self.data['date'], self.data['vix'], linewidth=1, alpha=0.8, color='blue')
        else:
            ax2.plot(self.data.index, self.data['vix'], linewidth=1, alpha=0.8, color='blue')
        
        # Highlight different regimes
        low_vol = self.data['vix'] < 20
        high_vol = self.data['vix'] > 30
        
        if 'date' in self.data.columns:
            ax2.fill_between(self.data['date'], 0, self.data['vix'].max()*1.1, 
                           where=low_vol, alpha=0.2, color='green', label='Low Vol (<20)')
            ax2.fill_between(self.data['date'], 0, self.data['vix'].max()*1.1, 
                           where=high_vol, alpha=0.2, color='red', label='High Vol (>30)')
        else:
            ax2.fill_between(self.data.index, 0, self.data['vix'].max()*1.1, 
                           where=low_vol, alpha=0.2, color='green', label='Low Vol (<20)')
            ax2.fill_between(self.data.index, 0, self.data['vix'].max()*1.1, 
                           where=high_vol, alpha=0.2, color='red', label='High Vol (>30)')
        
        ax2.set_title('VIX Time Series with Regime Highlighting')
        ax2.set_ylabel('VIX')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution plots for regime identification
        ax3 = axes[1, 0]
        
        # VIX distribution with regime boundaries
        ax3.hist(self.data['vix'], bins=50, alpha=0.7, edgecolor='black', density=True)
        ax3.axvline(x=20, color='orange', linestyle='--', linewidth=2, label='VIX=20')
        ax3.axvline(x=30, color='red', linestyle='--', linewidth=2, label='VIX=30')
        ax3.axvline(x=self.data['vix'].median(), color='blue', linestyle='-', linewidth=2, label='Median')
        
        ax3.set_title('VIX Distribution with Regime Boundaries')
        ax3.set_xlabel('VIX')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Regime transition analysis
        ax4 = axes[1, 1]
        
        # Create simple regime labels
        regime_labels = pd.cut(self.data['vix'], bins=[0, 15, 25, 40, 100], 
                              labels=['Low', 'Medium', 'High', 'Extreme'])
        
        # Count regime transitions
        regime_changes = regime_labels != regime_labels.shift(1)
        transition_dates = self.data.loc[regime_changes, 'date'] if 'date' in self.data.columns else self.data.index[regime_changes]
        
        # Plot regime over time
        regime_numeric = regime_labels.cat.codes
        if 'date' in self.data.columns:
            ax4.plot(self.data['date'], regime_numeric, linewidth=2, alpha=0.8)
        else:
            ax4.plot(self.data.index, regime_numeric, linewidth=2, alpha=0.8)
        
        ax4.set_title(f'Market Regime Over Time\n({regime_changes.sum()} regime transitions)')
        ax4.set_ylabel('Regime')
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(['Low', 'Medium', 'High', 'Extreme'])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self):
        """
        Single comprehensive dashboard
        """
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Financial Data Analysis Dashboard', fontsize=20)
        
        financial_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        
        if not available_cols:
            print("No financial data available for dashboard")
            return None
        
        # 1. Main time series (top row, spans 3 columns)
        ax1 = fig.add_subplot(gs[0, :3])
        for col in available_cols:
            if 'date' in self.data.columns:
                ax1.plot(self.data['date'], self.data[col], label=col.title(), alpha=0.8)
            else:
                ax1.plot(self.data.index, self.data[col], label=col.title(), alpha=0.8)
        
        ax1.set_title('Financial Time Series Overview')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Correlation heatmap (top right)
        ax2 = fig.add_subplot(gs[0, 3])
        if len(available_cols) > 1:
            corr_matrix = self.data[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax2, fmt='.2f', cbar_kws={'shrink': 0.8})
            ax2.set_title('Correlations')
        
        # 3. VIX analysis (second row, left)
        if 'vix' in available_cols:
            ax3 = fig.add_subplot(gs[1, :2])
            ax3.hist(self.data['vix'], bins=30, alpha=0.7, edgecolor='black')
            ax3.axvline(x=20, color='orange', linestyle='--', label='VIX=20')
            ax3.axvline(x=30, color='red', linestyle='--', label='VIX=30')
            ax3.set_title('VIX Distribution')
            ax3.set_xlabel('VIX')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # VIX regime over time
            ax4 = fig.add_subplot(gs[1, 2:])
            regime_colors = ['green' if v < 20 else 'orange' if v < 30 else 'red' for v in self.data['vix']]
            if 'date' in self.data.columns:
                ax4.scatter(self.data['date'], self.data['vix'], c=regime_colors, alpha=0.6, s=10)
            else:
                ax4.scatter(self.data.index, self.data['vix'], c=regime_colors, alpha=0.6, s=10)
            ax4.set_title('VIX Regime Over Time')
            ax4.set_ylabel('VIX')
        
        # 4. Yield curve analysis (third row)
        if 'treasury_10y' in available_cols and 'treasury_13w' in available_cols:
            yield_slope = self.data['treasury_10y'] - self.data['treasury_13w']
            
            ax5 = fig.add_subplot(gs[2, :2])
            if 'date' in self.data.columns:
                ax5.plot(self.data['date'], yield_slope, alpha=0.8)
            else:
                ax5.plot(self.data.index, yield_slope, alpha=0.8)
            ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax5.set_title('Yield Curve Slope (10Y - 13W)')
            ax5.set_ylabel('Spread (%)')
            ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(gs[2, 2:])
            ax6.hist(yield_slope, bins=30, alpha=0.7, edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', label='Flat Curve')
            ax6.set_title('Yield Slope Distribution')
            ax6.set_xlabel('Spread (%)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 5. Data quality indicators (bottom row)
        ax7 = fig.add_subplot(gs[3, :2])
        missing_data = self.data[available_cols].isnull().sum()
        missing_data.plot(kind='bar', ax=ax7, alpha=0.7)
        ax7.set_title('Missing Values by Variable')
        ax7.set_ylabel('Count')
        ax7.tick_params(axis='x', rotation=45)
        
        # Summary statistics table
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        
        # Create summary statistics
        summary_stats = self.data[available_cols].describe().round(3)
        table_data = []
        for col in summary_stats.columns:
            table_data.append([
                col.title(),
                f"{summary_stats.loc['mean', col]:.2f}",
                f"{summary_stats.loc['std', col]:.2f}",
                f"{summary_stats.loc['min', col]:.2f}",
                f"{summary_stats.loc['max', col]:.2f}"
            ])
        
        table = ax8.table(cellText=table_data,
                         colLabels=['Variable', 'Mean', 'Std', 'Min', 'Max'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax8.set_title('Summary Statistics', y=0.8)
        
        return fig
    
    def create_feature_analysis_plots(self):
        """
        Analyze created features if available
        """
        if self.features.empty:
            print("No features available for analysis")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Analysis Dashboard', fontsize=16)
        
        # 1. Feature correlation heatmap
        ax1 = axes[0, 0]
        # Select a subset of features for visualization
        feature_cols = self.features.columns[:20]  # First 20 features
        if len(feature_cols) > 1:
            feature_corr = self.features[feature_cols].corr()
            sns.heatmap(feature_corr, ax=ax1, cmap='coolwarm', center=0, 
                       square=True, cbar_kws={'shrink': 0.8})
            ax1.set_title('Feature Correlations (subset)')
        
        # 2. Feature importance based on variance
        ax2 = axes[0, 1]
        feature_vars = self.features.var().sort_values(ascending=False)[:15]
        feature_vars.plot(kind='bar', ax=ax2, alpha=0.7)
        ax2.set_title('Top 15 Features by Variance')
        ax2.set_ylabel('Variance')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Missing values in features
        ax3 = axes[1, 0]
        missing_features = self.features.isnull().sum()
        missing_features = missing_features[missing_features > 0].sort_values(ascending=False)
        if len(missing_features) > 0:
            missing_features[:10].plot(kind='bar', ax=ax3, alpha=0.7, color='red')
            ax3.set_title('Top 10 Features with Missing Values')
            ax3.set_ylabel('Missing Count')
        else:
            ax3.text(0.5, 0.5, 'No Missing Values\nin Features', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=14)
            ax3.set_title('Missing Values in Features')
        
        # 4. Feature distribution examples
        ax4 = axes[1, 1]
        # Plot distribution of a few key features
        key_features = [col for col in self.features.columns 
                       if any(keyword in col for keyword in ['return', 'momentum', 'vol', 'slope'])][:3]
        
        for i, feature in enumerate(key_features):
            ax4.hist(self.features[feature].dropna(), bins=30, alpha=0.5, 
                    label=feature, density=True)
        
        if key_features:
            ax4.set_title('Sample Feature Distributions')
            ax4.set_xlabel('Value')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig