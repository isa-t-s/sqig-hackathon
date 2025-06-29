import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

class FinancialDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.clean_data = None

    def load_raw_data(self):
        """
        Load your 2000 data points
        Expected columns: ['date', 'vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        """
        try:
            # Determine file type and load accordingly
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path)
            elif self.data_path.endswith(('.xlsx', '.xls')):
                self.raw_data = pd.read_excel(self.data_path)
            else:
                # Try CSV as default
                self.raw_data = pd.read_csv(self.data_path)
            
            print(f"Successfully loaded {len(self.raw_data)} rows from {self.data_path}")
            
            # Handle different column naming conventions
            self._standardize_column_names()
            
            # Convert date column to datetime
            self._process_date_column()
            
            # Ensure proper data types
            self._convert_data_types()
            
            # Sort by date
            if 'date' in self.raw_data.columns:
                self.raw_data = self.raw_data.sort_values('date').reset_index(drop=True)
            
            # Create clean_data copy
            self.clean_data = self.raw_data.copy()
            
            print("Data loading completed successfully!")
            return self.raw_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _standardize_column_names(self):
        """
        Standardize column names to expected format
        """
        # Common column name variations
        column_mappings = {
            # Date variations
            'Date': 'date', 'DATE': 'date', 'timestamp': 'date', 'Timestamp': 'date',
            
            # VIX variations
            'VIX': 'vix',
            
            # Treasury 13W variations
            'T-Bill_13W_Yield': 'treasury_13w',
            
            # Treasury 10Y variations
            '10Y_Treasury_Yield': 'treasury_10y',
            
            # Credit spread variations
            'Credit_Spread': 'credit_spread',
        }
        
        # Apply mappings
        self.raw_data.rename(columns=column_mappings, inplace=True)
        
        # Infer date from position
        expected_cols = ['date', 'vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        if len(self.raw_data.columns) == 5 and not all(col in self.raw_data.columns for col in expected_cols):
            print("Inferring column names from position...")
            self.raw_data.columns = expected_cols
        
        print(f"Standardized columns: {list(self.raw_data.columns)}")
    
    def _process_date_column(self):
        """Process and validate date column"""
        if 'date' not in self.raw_data.columns:
            raise ValueError("No date column found in data")
        
        # Date format
        date_formats = ['%Y-%m-%d']
        
        for fmt in date_formats:
            try:
                self.raw_data['date'] = pd.to_datetime(self.raw_data['date'], format=fmt)
                print(f"Successfully parsed dates using format: {fmt}")
                break
            except:
                continue
        else:
            # If no format works, use pandas' flexible parsing
            try:
                self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
                print("Successfully parsed dates using pandas flexible parsing")
            except Exception as e:
                raise ValueError(f"Could not parse date column: {e}")
    
    def _convert_data_types(self):
        """Convert financial columns to appropriate numeric types"""
        numeric_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        
        for col in numeric_cols:
            if col in self.raw_data.columns:
                # Convert to numeric, coercing errors to NaN
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
        
        print("Data types converted successfully")

    def validate_data_integrity(self):
        """
        Check data quality and completeness
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        print("=== DATA INTEGRITY VALIDATION ===")
        validation_results = {}
        
        # Check for missing values
        print("\n1. Missing Values Analysis:")
        missing_data = self.raw_data.isnull().sum()
        missing_percent = (missing_data / len(self.raw_data)) * 100
        
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent.round(2)
        })
        print(missing_summary)
        validation_results['missing_values'] = missing_summary
        
        # Check for duplicate dates
        print("\n2. Duplicate Dates Check:")
        if 'date' in self.raw_data.columns:
            duplicate_dates = self.raw_data['date'].duplicated().sum()
            print(f"Duplicate dates found: {duplicate_dates}")
            if duplicate_dates > 0:
                print("Sample duplicate dates:")
                print(self.raw_data[self.raw_data['date'].duplicated(keep=False)]['date'].head())
            validation_results['duplicate_dates'] = duplicate_dates
        
        # Check date continuity
        print("\n3. Date Continuity Check:")
        if 'date' in self.raw_data.columns and len(self.raw_data) > 1:
            date_gaps = self._check_date_gaps()
            validation_results['date_gaps'] = date_gaps
        
        # Validate financial ranges
        print("\n4. Financial Range Validation:")
        range_issues = self._validate_financial_ranges()
        validation_results['range_issues'] = range_issues
        
        # Check for outliers
        print("\n5. Outlier Detection:")
        outlier_summary = self._detect_outliers()
        validation_results['outliers'] = outlier_summary
        
        # Data types validation
        print("\n6. Data Types:")
        print(self.raw_data.dtypes)
        validation_results['data_types'] = dict(self.raw_data.dtypes)
        
        return validation_results
    
    def _check_date_gaps(self):
        """Check for gaps in date sequence"""
        dates = pd.to_datetime(self.raw_data['date']).sort_values()
        date_gaps = []
        
        for i in range(1, len(dates)):
            gap = (dates.iloc[i] - dates.iloc[i-1]).days
            if gap > 7:  # Gaps larger than 7 days
                date_gaps.append({
                    'start_date': dates.iloc[i-1],
                    'end_date': dates.iloc[i],
                    'gap_days': gap
                })
        
        if date_gaps:
            print(f"Found {len(date_gaps)} significant date gaps (>7 days):")
            for gap in date_gaps[:3]:  # Show first 3
                print(f"  {gap['start_date'].date()} to {gap['end_date'].date()}: {gap['gap_days']} days")
        else:
            print("No significant date gaps found")
        
        return date_gaps
    
    def _validate_financial_ranges(self):
        """Validate that financial values are within realistic ranges"""
        range_checks = {
            'vix': {'min': 0, 'max': 200, 'typical_min': 8, 'typical_max': 80},
            'treasury_13w': {'min': -2, 'max': 20, 'typical_min': 0, 'typical_max': 10},
            'treasury_10y': {'min': -2, 'max': 20, 'typical_min': 0, 'typical_max': 10},
            'credit_spread': {'min': -100, 'max': 5000, 'typical_min': 0, 'typical_max': 1000}
        }
        
        range_issues = {}
        
        for col, ranges in range_checks.items():
            if col in self.raw_data.columns:
                data_col = self.raw_data[col].dropna()
                
                # Count impossible values
                impossible_low = (data_col < ranges['min']).sum()
                impossible_high = (data_col > ranges['max']).sum()
                
                # Count atypical values
                atypical_low = (data_col < ranges['typical_min']).sum()
                atypical_high = (data_col > ranges['typical_max']).sum()
                
                range_issues[col] = {
                    'range': f"{data_col.min():.2f} to {data_col.max():.2f}",
                    'impossible_low': impossible_low,
                    'impossible_high': impossible_high,
                    'atypical_low': atypical_low,
                    'atypical_high': atypical_high
                }
                
                print(f"{col}: {data_col.min():.2f} to {data_col.max():.2f}")
                if impossible_low > 0:
                    print(f"  WARNING: {impossible_low} values below minimum threshold")
                if impossible_high > 0:
                    print(f"  WARNING: {impossible_high} values above maximum threshold")
        
        return range_issues
    
    def _detect_outliers(self):
        """Detect outliers using IQR method"""
        numeric_columns = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        outlier_summary = {}
        
        for col in numeric_columns:
            if col in self.raw_data.columns:
                data_col = self.raw_data[col].dropna()
                Q1 = data_col.quantile(0.25)
                Q3 = data_col.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((data_col < lower_bound) | (data_col > upper_bound)).sum()
                outlier_summary[col] = outliers
                
                if outliers > 0:
                    print(f"{col}: {outliers} outliers detected")
        
        return outlier_summary

    def basic_data_info(self):
        """
        Generate summary statistics and info
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        print("=== BASIC DATA INFORMATION ===")
        info_summary = {}
        
        # 1. Dataset overview
        print(f"\n1. Dataset Overview:")
        shape_info = {
            'rows': self.raw_data.shape[0],
            'columns': self.raw_data.shape[1],
            'memory_kb': self.raw_data.memory_usage(deep=True).sum() / 1024
        }
        
        print(f"   Shape: {shape_info['rows']} rows × {shape_info['columns']} columns")
        print(f"   Memory usage: {shape_info['memory_kb']:.2f} KB")
        
        if 'date' in self.raw_data.columns:
            date_range = {
                'start': self.raw_data['date'].min(),
                'end': self.raw_data['date'].max(),
                'span_days': (self.raw_data['date'].max() - self.raw_data['date'].min()).days
            }
            print(f"   Date range: {date_range['start'].date()} to {date_range['end'].date()}")
            print(f"   Time span: {date_range['span_days']} days")
            info_summary['date_range'] = date_range
        
        info_summary['shape'] = shape_info
        
        # Descriptive statistics
        print(f"\n2. Descriptive Statistics:")
        numeric_columns = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']
        available_numeric = [col for col in numeric_columns if col in self.raw_data.columns]
        
        if available_numeric:
            stats = self.raw_data[available_numeric].describe()
            print(stats.round(4))
            info_summary['statistics'] = stats
        
        # Data completeness
        print(f"\n3. Data Completeness:")
        completeness = ((len(self.raw_data) - self.raw_data.isnull().sum()) / 
                       len(self.raw_data) * 100)
        for col in self.raw_data.columns:
            print(f"   {col}: {completeness[col]:.1f}% complete")
        info_summary['completeness'] = completeness
        
        # Financial metrics insights
        print(f"\n4. Financial Metrics Insights:")
        financial_insights = {}
        
        if 'vix' in self.raw_data.columns:
            vix_high = (self.raw_data['vix'] > 30).sum()
            vix_extreme = (self.raw_data['vix'] > 40).sum()
            vix_insights = {
                'high_volatility_days': vix_high,
                'extreme_volatility_days': vix_extreme,
                'high_vol_pct': vix_high/len(self.raw_data)*100,
                'extreme_vol_pct': vix_extreme/len(self.raw_data)*100
            }
            print(f"   VIX > 30 (high volatility): {vix_high} days ({vix_insights['high_vol_pct']:.1f}%)")
            print(f"   VIX > 40 (extreme volatility): {vix_extreme} days ({vix_insights['extreme_vol_pct']:.1f}%)")
            financial_insights['vix'] = vix_insights
        
        if 'treasury_13w' in self.raw_data.columns and 'treasury_10y' in self.raw_data.columns:
            # Yield curve analysis
            yield_curve = self.raw_data['treasury_10y'] - self.raw_data['treasury_13w']
            inverted_curve = (yield_curve < 0).sum()
            curve_insights = {
                'inverted_days': inverted_curve,
                'inverted_pct': inverted_curve/len(self.raw_data)*100,
                'avg_slope': yield_curve.mean()
            }
            print(f"   Inverted yield curve days: {inverted_curve} ({curve_insights['inverted_pct']:.1f}%)")
            financial_insights['yield_curve'] = curve_insights
        
        if 'credit_spread' in self.raw_data.columns:
            high_spread_threshold = self.raw_data['credit_spread'].quantile(0.75)
            high_spread = (self.raw_data['credit_spread'] > high_spread_threshold).sum()
            spread_insights = {
                'high_spread_days': high_spread,
                'high_spread_threshold': high_spread_threshold,
                'avg_spread': self.raw_data['credit_spread'].mean()
            }
            print(f"   High credit spread periods (>75th percentile): {high_spread} days")
            financial_insights['credit_spread'] = spread_insights
        
        info_summary['financial_insights'] = financial_insights
        
        # Correlation matrix
        print(f"\n5. Correlation Matrix:")
        if len(available_numeric) > 1:
            correlation_matrix = self.raw_data[available_numeric].corr()
            print(correlation_matrix.round(3))
            info_summary['correlations'] = correlation_matrix
        
        # Data preview
        print(f"\n6. Data Preview:")
        print("First 3 rows:")
        print(self.raw_data.head(3))
        print("\nLast 3 rows:")
        print(self.raw_data.tail(3))
        
        return info_summary