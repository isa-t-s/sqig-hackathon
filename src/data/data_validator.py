import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class DataValidator:
    def __init__(self, data):
        self.data = data
        self.validation_report = {}
    
    def check_date_consistency(self):
        """
        Validate date column and identify gaps
        """
        if 'date' not in self.data.columns:
            return {"error": "No date column found"}
        
        dates = pd.to_datetime(self.data['date']).sort_values()
        
        # Check for date gaps > 3 business days
        business_day_gaps = []
        weekend_holiday_flags = []
        irregular_patterns = []
        
        for i in range(1, len(dates)):
            gap = (dates.iloc[i] - dates.iloc[i-1]).days
            
            # Flag gaps > 3 business days (accounting for weekends)
            if gap > 3:
                business_day_gaps.append({
                    'start': dates.iloc[i-1],
                    'end': dates.iloc[i],
                    'gap_days': gap
                })
        
        # Check for weekends/holidays in data
        weekday_counts = dates.dt.dayofweek.value_counts().sort_index()
        
        # Detect irregular patterns
        expected_ratio = 1/5  # Each weekday should be ~20% if daily data
        for weekday, count in weekday_counts.items():
            actual_ratio = count / len(dates)
            if abs(actual_ratio - expected_ratio) > 0.1:  # 10% tolerance
                irregular_patterns.append({
                    'weekday': weekday,
                    'count': count,
                    'ratio': actual_ratio,
                    'expected_ratio': expected_ratio
                })
        
        results = {
            'business_day_gaps': business_day_gaps,
            'weekday_distribution': weekday_counts.to_dict(),
            'irregular_patterns': irregular_patterns,
            'total_dates': len(dates),
            'unique_dates': dates.nunique(),
            'date_range': (dates.min(), dates.max())
        }
        
        self.validation_report['date_consistency'] = results
        return results
        
    def validate_financial_ranges(self):
        """
        Check if values are within realistic financial ranges
        """
        # Define realistic ranges for financial instruments
        range_definitions = {
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
        
        validation_results = {}
        
        for col, ranges in range_definitions.items():
            if col in self.data.columns:
                data_col = self.data[col].dropna()
                
                if len(data_col) == 0:
                    validation_results[col] = {'error': 'No valid data'}
                    continue
                
                # Count violations
                absolute_violations = {
                    'below_min': (data_col < ranges['absolute_min']).sum(),
                    'above_max': (data_col > ranges['absolute_max']).sum()
                }
                
                typical_violations = {
                    'below_typical': (data_col < ranges['typical_min']).sum(),
                    'above_typical': (data_col > ranges['typical_max']).sum()
                }
                
                extreme_violations = {
                    'below_extreme': (data_col < ranges['extreme_min']).sum(),
                    'above_extreme': (data_col > ranges['extreme_max']).sum()
                }
                
                # Calculate statistics
                stats_summary = {
                    'min': data_col.min(),
                    'max': data_col.max(),
                    'mean': data_col.mean(),
                    'std': data_col.std(),
                    'q1': data_col.quantile(0.25),
                    'median': data_col.median(),
                    'q3': data_col.quantile(0.75)
                }
                
                validation_results[col] = {
                    'description': ranges['description'],
                    'statistics': stats_summary,
                    'absolute_violations': absolute_violations,
                    'typical_violations': typical_violations,
                    'extreme_violations': extreme_violations,
                    'total_points': len(data_col),
                    'ranges': ranges
                }
        
        self.validation_report['financial_ranges'] = validation_results
        return validation_results
        
    def detect_structural_breaks(self):
        """
        Identify potential data issues or regime changes
        """
        structural_break_results = {}
        
        numeric_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        for col in available_cols:
            data_col = self.data[col].dropna()
            
            if len(data_col) < 50:  # Need minimum data points
                structural_break_results[col] = {'error': 'Insufficient data for analysis'}
                continue
            
            # Rolling mean and std to detect sudden changes
            window = min(30, len(data_col) // 4)  # Adaptive window size
            rolling_mean = data_col.rolling(window).mean()
            rolling_std = data_col.rolling(window).std()
            
            # Detect large jumps in rolling mean
            mean_changes = rolling_mean.diff().abs()
            mean_threshold = rolling_mean.std() * 2  # 2 standard deviations
            large_mean_jumps = mean_changes > mean_threshold
            
            # Detect large changes in rolling std (volatility regime changes)
            std_changes = rolling_std.diff().abs()
            std_threshold = rolling_std.std() * 2
            large_std_jumps = std_changes > std_threshold
            
            # Simple structural break test using rolling statistics
            # Split data into chunks and compare means
            n_chunks = 5
            chunk_size = len(data_col) // n_chunks
            chunk_means = []
            chunk_stds = []
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(data_col)
                chunk_data = data_col.iloc[start_idx:end_idx]
                chunk_means.append(chunk_data.mean())
                chunk_stds.append(chunk_data.std())
            
            # Test for significant differences between chunks
            chunk_mean_cv = np.std(chunk_means) / np.mean(chunk_means) if np.mean(chunk_means) != 0 else 0
            chunk_std_cv = np.std(chunk_stds) / np.mean(chunk_stds) if np.mean(chunk_stds) != 0 else 0
            
            # Z-score analysis for outliers
            z_scores = np.abs(stats.zscore(data_col))
            extreme_outliers = (z_scores > 3).sum()
            moderate_outliers = ((z_scores > 2) & (z_scores <= 3)).sum()
            
            structural_break_results[col] = {
                'rolling_analysis': {
                    'large_mean_jumps': large_mean_jumps.sum(),
                    'large_std_jumps': large_std_jumps.sum(),
                    'mean_jump_dates': data_col.index[large_mean_jumps].tolist()[:5],  # Top 5
                    'std_jump_dates': data_col.index[large_std_jumps].tolist()[:5]    # Top 5
                },
                'chunk_analysis': {
                    'chunk_means': chunk_means,
                    'chunk_stds': chunk_stds,
                    'mean_coefficient_variation': chunk_mean_cv,
                    'std_coefficient_variation': chunk_std_cv
                },
                'outlier_analysis': {
                    'extreme_outliers': extreme_outliers,
                    'moderate_outliers': moderate_outliers,
                    'total_data_points': len(data_col)
                }
            }
        
        self.validation_report['structural_breaks'] = structural_break_results
        return structural_break_results
    
    def create_validation_report(self):
        """
        Generate data quality report
        """
        print("=== DATA VALIDATION REPORT ===")
        
        # Run all validation checks
        date_results = self.check_date_consistency()
        range_results = self.validate_financial_ranges()
        structural_results = self.detect_structural_breaks()
        
        # Generate summary
        total_issues = 0
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Analyze date consistency issues
        if 'error' not in date_results:
            if len(date_results['business_day_gaps']) > 0:
                total_issues += len(date_results['business_day_gaps'])
                warnings.append(f"Found {len(date_results['business_day_gaps'])} significant date gaps")
            
            if len(date_results['irregular_patterns']) > 0:
                warnings.append("Irregular weekday distribution detected")
        
        # Analyze financial range issues
        for col, results in range_results.items():
            if 'error' not in results:
                abs_violations = (results['absolute_violations']['below_min'] + 
                                results['absolute_violations']['above_max'])
                if abs_violations > 0:
                    critical_issues.append(f"{col}: {abs_violations} values outside absolute bounds")
                    total_issues += abs_violations
        
        # Analyze structural breaks
        for col, results in structural_results.items():
            if 'error' not in results:
                if results['rolling_analysis']['large_mean_jumps'] > 3:
                    warnings.append(f"{col}: Multiple large mean jumps detected")
                if results['outlier_analysis']['extreme_outliers'] > len(self.data) * 0.01:  # >1% outliers
                    warnings.append(f"{col}: High number of extreme outliers")
        
        # Generate recommendations
        if total_issues == 0:
            recommendations.append("Data quality looks good!")
        else:
            if critical_issues:
                recommendations.append("CRITICAL: Review and clean data with absolute bound violations")
            if len(warnings) > 3:
                recommendations.append("Consider additional data preprocessing")
            recommendations.append("Document all data quality issues for model interpretation")
        
        # Create final report
        final_report = {
            'summary': {
                'total_issues': total_issues,
                'critical_issues': len(critical_issues),
                'warnings': len(warnings),
                'data_shape': self.data.shape,
                'validation_timestamp': pd.Timestamp.now()
            },
            'critical_issues': critical_issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'detailed_results': {
                'date_consistency': date_results,
                'financial_ranges': range_results,
                'structural_breaks': structural_results
            }
        }
        
        # Print summary
        print(f"\nValidation Summary:")
        print(f"  Total Issues: {total_issues}")
        print(f"  Critical Issues: {len(critical_issues)}")
        print(f"  Warnings: {len(warnings)}")
        
        if critical_issues:
            print(f"\nCritical Issues:")
            for issue in critical_issues:
                print(f"  - {issue}")
        
        if warnings:
            print(f"\nWarnings:")
            for warning in warnings[:5]:  # Show top 5
                print(f"  - {warning}")
        
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        
        self.validation_report['final_report'] = final_report
        return final_report
    
    def create_validation_visualizations(self):
        """Create visualizations for data quality issues"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality Validation Dashboard', fontsize=16)
        
        numeric_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        if len(available_cols) >= 1:
            # Time series plot with outliers highlighted
            ax1 = axes[0, 0]
            col = available_cols[0]
            data_col = self.data[col].dropna()
            z_scores = np.abs(stats.zscore(data_col))
            outliers = z_scores > 3
            
            ax1.plot(range(len(data_col)), data_col, alpha=0.7, label='Data')
            ax1.scatter(np.where(outliers)[0], data_col[outliers], 
                       color='red', s=20, label='Outliers (Z>3)')
            ax1.set_title(f'{col.title()} with Outliers')
            ax1.legend()
            
            # Distribution plot
            ax2 = axes[0, 1]
            ax2.hist(data_col, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(data_col.mean(), color='red', linestyle='--', label='Mean')
            ax2.axvline(data_col.median(), color='green', linestyle='--', label='Median')
            ax2.set_title(f'{col.title()} Distribution')
            ax2.legend()
        
        if len(available_cols) >= 2:
            # Correlation heatmap
            ax3 = axes[1, 0]
            corr_data = self.data[available_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Correlation Matrix')
            
            # Missing values heatmap
            ax4 = axes[1, 1]
            missing_data = self.data[available_cols].isnull()
            if missing_data.sum().sum() > 0:
                sns.heatmap(missing_data, cmap='viridis', ax=ax4)
                ax4.set_title('Missing Values Pattern')
            else:
                ax4.text(0.5, 0.5, 'No Missing Values', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('Missing Values Pattern')
        
        plt.tight_layout()
        return fig