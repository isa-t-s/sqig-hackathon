# view_plots.py - Multiple ways to view your generated plots

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import webbrowser

def show_plots_in_matplotlib():
    """Display all plots using matplotlib"""
    
    viz_dir = 'data/processed/visualizations'
    
    if not os.path.exists(viz_dir):
        print(f"❌ Visualization directory not found: {viz_dir}")
        print("   Make sure the pipeline ran successfully first")
        return
    
    # Find all PNG files
    plot_files = list(Path(viz_dir).glob('*.png'))
    
    if not plot_files:
        print(f"❌ No plot files found in {viz_dir}")
        return
    
    print(f"📊 Found {len(plot_files)} plots to display...")
    
    # Display each plot
    for i, plot_file in enumerate(plot_files, 1):
        print(f"\n{i}. Displaying: {plot_file.name}")
        
        # Create a new figure
        plt.figure(figsize=(12, 8))
        
        # Load and display image
        img = mpimg.imread(plot_file)
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        plt.title(plot_file.stem.replace('_', ' ').title(), fontsize=14, pad=20)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Ask user if they want to continue
        if i < len(plot_files):
            response = input(f"\nPress Enter to continue to next plot (or 'q' to quit): ")
            if response.lower() == 'q':
                break

def show_plots_in_grid():
    """Display all plots in a grid layout"""
    
    viz_dir = 'data/processed/visualizations'
    plot_files = list(Path(viz_dir).glob('*.png'))
    
    if not plot_files:
        print(f"❌ No plot files found in {viz_dir}")
        return
    
    # Calculate grid size
    n_plots = len(plot_files)
    cols = 2
    rows = (n_plots + cols - 1) // cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    
    # Handle case where we have only one row
    if rows == 1:
        axes = [axes] if n_plots == 1 else axes
    elif n_plots == 1:
        axes = [axes[0, 0]]
    else:
        axes = axes.flatten()
    
    # Display each plot
    for i, plot_file in enumerate(plot_files):
        img = mpimg.imread(plot_file)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(plot_file.stem.replace('_', ' ').title(), fontsize=12)
    
    # Hide any unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def list_available_plots():
    """List all available plots with descriptions"""
    
    viz_dir = 'data/processed/visualizations'
    
    if not os.path.exists(viz_dir):
        print(f"❌ Visualization directory not found: {viz_dir}")
        return
    
    plot_files = list(Path(viz_dir).glob('*.png'))
    
    if not plot_files:
        print(f"❌ No plot files found in {viz_dir}")
        return
    
    # Plot descriptions
    descriptions = {
        'time_series_analysis': 'Financial time series with moving averages and trend analysis',
        'correlation_analysis': 'Correlation matrices and rolling correlation analysis',
        'regime_identification': 'VIX vs yield curve analysis and regime period visualization', 
        'summary_dashboard': 'Comprehensive overview dashboard with key metrics',
        'feature_analysis': 'Feature correlation and importance analysis',
        'regime_analysis_detailed': 'Detailed regime transition and duration analysis'
    }
    
    print("📊 Available Plots:")
    print("=" * 50)
    
    for i, plot_file in enumerate(plot_files, 1):
        file_stem = plot_file.stem
        description = descriptions.get(file_stem, 'Analysis visualization')
        file_size_kb = plot_file.stat().st_size // 1024
        
        print(f"{i}. {file_stem}.png")
        print(f"   📝 {description}")
        print(f"   📁 Size: {file_size_kb} KB")
        print(f"   📂 Path: {plot_file}")
        print()

def open_plot_folder():
    """Open the visualization folder in file explorer"""
    
    viz_dir = 'data/processed/visualizations'
    abs_path = os.path.abspath(viz_dir)
    
    if not os.path.exists(viz_dir):
        print(f"❌ Visualization directory not found: {viz_dir}")
        return
    
    try:
        # Windows
        if os.name == 'nt':
            os.startfile(abs_path)
        # macOS
        elif os.name == 'posix' and os.uname().sysname == 'Darwin':
            os.system(f'open "{abs_path}"')
        # Linux
        else:
            os.system(f'xdg-open "{abs_path}"')
        
        print(f"✅ Opened visualization folder: {abs_path}")
        
    except Exception as e:
        print(f"❌ Could not open folder: {e}")
        print(f"   Manual path: {abs_path}")

def show_specific_plot(plot_name):
    """Show a specific plot by name"""
    
    viz_dir = 'data/processed/visualizations'
    plot_path = os.path.join(viz_dir, f"{plot_name}.png")
    
    if not os.path.exists(plot_path):
        print(f"❌ Plot not found: {plot_path}")
        list_available_plots()
        return
    
    plt.figure(figsize=(12, 8))
    img = mpimg.imread(plot_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(plot_name.replace('_', ' ').title(), fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def create_html_viewer():
    """Create an HTML file to view all plots in browser"""
    
    viz_dir = 'data/processed/visualizations'
    plot_files = list(Path(viz_dir).glob('*.png'))
    
    if not plot_files:
        print(f"❌ No plot files found in {viz_dir}")
        return
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Phase 1 Analysis - Visualization Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .plot-container { margin: 30px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .plot-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .plot-description { color: #666; font-style: italic; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>📊 Phase 1 Financial Analysis - Visualization Dashboard</h1>
    <p><strong>Generated:</strong> """ + str(Path().absolute()) + """</p>
"""
    
    # Plot descriptions
    descriptions = {
        'time_series_analysis': 'Time series plots showing VIX, Treasury yields, and credit spreads over time with moving averages and trend analysis.',
        'correlation_analysis': 'Correlation matrices showing relationships between financial variables, including static and rolling correlations.',
        'regime_identification': 'Regime analysis showing VIX vs yield curve relationships and identification of market regime periods.',
        'summary_dashboard': 'Comprehensive dashboard combining key metrics, distributions, and summary statistics.',
        'feature_analysis': 'Analysis of engineered features including correlations, variance, and missing value patterns.',
        'regime_analysis_detailed': 'Detailed regime analysis showing transitions, durations, and regime evolution over time.'
    }
    
    # Add each plot
    for plot_file in sorted(plot_files):
        file_stem = plot_file.stem
        description = descriptions.get(file_stem, 'Financial analysis visualization')
        
        html_content += f"""
    <div class="plot-container">
        <h2>{file_stem.replace('_', ' ').title()}</h2>
        <img src="visualizations/{plot_file.name}" alt="{file_stem}" class="plot-image">
        <p class="plot-description">{description}</p>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    html_path = 'data/processed/visualization_dashboard.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Created HTML dashboard: {html_path}")
    
    # Try to open in browser
    try:
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
        print("✅ Opened dashboard in browser")
    except:
        print(f"   Open manually: {os.path.abspath(html_path)}")

def interactive_plot_viewer():
    """Interactive menu for viewing plots"""
    
    while True:
        print("\n" + "="*50)
        print("📊 PLOT VIEWER - Choose an option:")
        print("="*50)
        print("1. List all available plots")
        print("2. Show all plots one by one")
        print("3. Show all plots in grid layout")
        print("4. Show specific plot")
        print("5. Open visualization folder")
        print("6. Create HTML dashboard")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            list_available_plots()
        elif choice == '2':
            show_plots_in_matplotlib()
        elif choice == '3':
            show_plots_in_grid()
        elif choice == '4':
            list_available_plots()
            plot_name = input("\nEnter plot name (without .png): ").strip()
            show_specific_plot(plot_name)
        elif choice == '5':
            open_plot_folder()
        elif choice == '6':
            create_html_viewer()
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    interactive_plot_viewer()