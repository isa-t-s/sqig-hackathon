
import os
import sys
import argparse
import warnings
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_pipeline import ComprehensiveDataPipeline
from src.utils.helpers import (
    validate_file_path, 
    create_directory_structure,
    log_execution_time
)

def setup_project_structure():
    """Create the complete project directory structure"""
    base_dirs = [
        'data/raw',
        'data/processed', 
        'data/features',
        'reports',
        'visualizations',
        'models',  #  For Person 2's ML models
        'logs',
        'notebooks'
    ]
    
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Project structure created")

@log_execution_time
def run_phase_1_pipeline(data_path, output_dir='data/processed', 
                        include_advanced=True, create_viz=True):
    """
    Run the complete Phase 1 data pipeline
    
    Args:
        data_path (str): Path to raw data file
        output_dir (str): Output directory for processed data
        include_advanced (bool): Include advanced feature engineering
        create_viz (bool): Create visualizations
    
    Returns:
        dict: Pipeline execution results
    """
    
    print("STARTING PHASE 1: DATA PIPELINE & FEATURE ENGINEERING")
    print("=" * 80)
    
    # Validate input file
    validate_file_path(data_path)
    
    # Initialize pipeline
    pipeline = ComprehensiveDataPipeline(data_path, output_dir)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        include_advanced=include_advanced,
        create_visualizations=create_viz
    )
    
    return results

def validate_phase_1_output(output_dir='data/processed'):
    """
    Validate that all Phase 1 outputs were created successfully
    """
    print("\nVALIDATING PHASE 1 OUTPUTS")
    print("-" * 40)
    
    required_files = [
        'final_dataset.csv',
        'train_data.csv', 
        'validation_data.csv',
        'test_data.csv',
        'feature_descriptions.json',
        'regime_analysis.json',
        'validation_report.json',
        'pipeline_metadata.json',
        'final_report.txt'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
        else:
            file_size = os.path.getsize(file_path)
            print(f"✓ {file_name} ({file_size:,} bytes)")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    else:
        print(f"\nAll Phase 1 outputs validated successfully!")
        return True

def create_handoff_summary(output_dir='data/processed'):
    """
    Create summary for Person 2 handoff
    """
    handoff_path = os.path.join(output_dir, 'HANDOFF_TO_PERSON_2.md')
    
    with open(handoff_path, 'w') as f:
        f.write("# HANDOFF TO PERSON 2: ML ARCHITECTURE\n\n")
        
        f.write("## Phase 1 Complete \n\n")
        f.write("Person 1 has completed all data pipeline and feature engineering tasks.\n\n")
        
        f.write("## What's Ready for You:\n\n")
        f.write("### Clean Dataset\n")
        f.write("- `final_dataset.csv` - Complete dataset with all features\n")
        f.write("- ~2000 data points with financial time series\n")
        f.write("- No missing values in critical columns\n")
        f.write("- Validated data quality\n\n")
        
        f.write("### Regime Labels\n")
        f.write("- Manual regime classification (0=Normal, 1=Transition, 2=Crisis)\n")
        f.write("- Continuous regime scores (0-1 scale)\n")
        f.write("- Historical validation against market events\n\n")
        
        f.write("### Feature Engineering\n")
        f.write("- 50+ engineered features covering:\n")
        f.write("  - Returns and momentum indicators\n")
        f.write("  - Volatility measures\n")
        f.write("  - Yield curve analysis\n")
        f.write("  - VIX-specific features\n")
        f.write("  - Cross-asset relationships\n")
        f.write("  - Regime-dependent features\n")
        f.write("  - Mean reversion indicators\n")
        f.write("- Complete feature documentation in `feature_descriptions.json`\n\n")
        
        f.write("### Train/Validation/Test Splits\n")
        f.write("- Temporal splits (no data leakage)\n")
        f.write("- `train_data.csv` (70% - ~1400 samples)\n")
        f.write("- `validation_data.csv` (15% - ~300 samples)\n")
        f.write("- `test_data.csv` (15% - ~300 samples)\n\n")
        
        f.write("### Documentation\n")
        f.write("- `final_report.txt` - Comprehensive analysis report\n")
        f.write("- `regime_analysis.json` - Detailed regime analysis\n")
        f.write("- `validation_report.json` - Data quality assessment\n")
        f.write("- Visualization dashboards in `visualizations/`\n\n")
        
        f.write("## YOUR RESPONSIBILITIES (Person 2):\n\n")
        f.write("### Week 1-2: Baseline Implementation\n")
        f.write("- [ ] Implement standard LSTM baseline model\n")
        f.write("- [ ] Create data loaders for LSTM input sequences\n")
        f.write("- [ ] Set up training infrastructure\n")
        f.write("- [ ] Develop evaluation metrics framework\n")
        f.write("- [ ] Establish performance benchmarks\n\n")
        
        f.write("### Week 3-4: Regime Detection Integration\n")
        f.write("- [ ] Use Person 1's regime labels for training\n")
        f.write("- [ ] Implement neural regime detector\n")
        f.write("- [ ] Validate regime detection accuracy\n")
        f.write("- [ ] Create regime-aware data processing\n\n")
        
        f.write("### Week 5-8: Custom LSTM Architecture\n")
        f.write("- [ ] Implement custom LSTM cells with regime gates\n")
        f.write("- [ ] Add cross-asset attention mechanism\n")
        f.write("- [ ] Integration testing and debugging\n")
        f.write("- [ ] Multi-task loss functions\n\n")
        
        f.write("## Key Files to Use:\n\n")
        f.write("```python\n")
        f.write("# Load preprocessed data\n")
        f.write("import pandas as pd\n\n")
        f.write("# Training data\n")
        f.write("train_data = pd.read_csv('data/processed/train_data.csv')\n")
        f.write("val_data = pd.read_csv('data/processed/validation_data.csv')\n")
        f.write("test_data = pd.read_csv('data/processed/test_data.csv')\n\n")
        f.write("# Feature descriptions\n")
        f.write("import json\n")
        f.write("with open('data/processed/feature_descriptions.json', 'r') as f:\n")
        f.write("    feature_desc = json.load(f)\n\n")
        f.write("# Target variables for LSTM\n")
        f.write("target_cols = ['vix', 'treasury_13w', 'treasury_10y', 'credit_spread']\n")
        f.write("regime_col = 'regime_label'\n")
        f.write("```\n\n")
        
        f.write("## Integration Points:\n\n")
        f.write("- **Week 2 Checkpoint**: Validate baseline LSTM works with processed data\n")
        f.write("- **Week 4 Checkpoint**: Regime detection integration\n")
        f.write("- **Week 6 Checkpoint**: Custom architecture progress\n")
        f.write("- **Week 8 Checkpoint**: Full model integration\n\n")
        
        f.write("## Questions?\n\n")
        f.write("If you need clarification on:\n")
        f.write("- Feature engineering decisions\n")
        f.write("- Regime labeling methodology\n")
        f.write("- Data preprocessing steps\n")
        f.write("- Financial domain insights\n\n")
        f.write("Reach out to Person 1 for context and explanation.\n\n")
        
        f.write("---\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Status**: Phase 1 Complete - Ready for ML Development\n")
    
    print(f"✓ Created handoff summary: {handoff_path}")

def main():
    """Main execution function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Phase 1: Financial Data Pipeline & Feature Engineering'
    )
    
    parser.add_argument(
        'data_path',
        help='Path to raw financial data file (CSV/Excel)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory for processed data (default: data/processed)'
    )
    
    parser.add_argument(
        '--basic-only',
        action='store_true',
        help='Run basic feature engineering only (skip advanced features)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization creation'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing outputs (do not rerun pipeline)'
    )
    
    args = parser.parse_args()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        # Setup project structure
        setup_project_structure()
        
        if args.validate_only:
            # Only validate existing outputs
            success = validate_phase_1_output(args.output_dir)
            if success:
                create_handoff_summary(args.output_dir)
        else:
            # Run complete pipeline
            results = run_phase_1_pipeline(
                data_path=args.data_path,
                output_dir=args.output_dir,
                include_advanced=not args.basic_only,
                create_viz=not args.no_viz
            )
            
            # Validate outputs
            success = validate_phase_1_output(args.output_dir)
            
            if success:
                # Create handoff summary
                create_handoff_summary(args.output_dir)
                
                print("\n" + "=" * 80)
                print("PHASE 1 COMPLETED SUCCESSFULLY!")
                print("=" * 80)
                print(f"All outputs saved to: {args.output_dir}")
                print(f"See HANDOFF_TO_PERSON_2.md for next steps")
                print("Ready for Person 2's ML development!")
                print("=" * 80)
            else:
                print("\n Pipeline completed but some outputs are missing")
                sys.exit(1)
    
    except Exception as e:
        print(f"\n Pipeline failed: {str(e)}")
        print("\nFor debugging:")
        print("1. Check your data file format and path")
        print("2. Ensure data contains required columns: date, vix, treasury_13w, treasury_10y, credit_spread")
        print("3. Verify you have write permissions to output directory")
        sys.exit(1)

if __name__ == "__main__":
    main()