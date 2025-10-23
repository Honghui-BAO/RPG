#!/usr/bin/env python3
"""
Analysis script for MHL model masking experiments.
Parses log files and generates performance comparison reports.
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def parse_log_file(log_file):
    """Parse a single log file and extract performance metrics."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract parameters from filename
        filename = os.path.basename(log_file)
        parts = filename.replace('.log', '').split('_')
        
        mask_ratio = None
        recon_weight = None
        category = parts[0] if len(parts) > 0 else 'Unknown'
        
        # Extract mask ratio and reconstruction weight from filename
        for part in parts:
            if part.startswith('mask_'):
                mask_ratio = float(part.split('_')[1])
            elif part.startswith('recon_'):
                recon_weight = float(part.split('_')[1])
        
        # Extract metrics from log content
        metrics = {
            'category': category,
            'mask_ratio': mask_ratio,
            'reconstruction_weight': recon_weight,
            'log_file': log_file
        }
        
        # Extract final test results
        test_results = re.search(r'Test Results: ({.*})', content)
        if test_results:
            # This would need to be adapted based on actual log format
            pass
        
        # Extract loss information
        loss_patterns = {
            'total_loss': r'Total loss: ([\d.]+)',
            'recommendation_loss': r'Recommendation loss: ([\d.]+)',
            'reconstruction_loss': r'Reconstruction loss: ([\d.]+)'
        }
        
        for metric, pattern in loss_patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[metric] = float(match.group(1))
        
        return metrics
        
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None

def analyze_results(results_dir):
    """Analyze all experiment results in the given directory."""
    log_files = glob.glob(os.path.join(results_dir, "*.log"))
    
    if not log_files:
        print(f"No log files found in {results_dir}")
        return None
    
    print(f"Found {len(log_files)} log files")
    
    # Parse all log files
    results = []
    for log_file in log_files:
        metrics = parse_log_file(log_file)
        if metrics:
            results.append(metrics)
    
    if not results:
        print("No valid results found")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate summary
    print("\n📊 Experiment Summary")
    print("=" * 50)
    print(f"Total experiments: {len(df)}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Mask ratios: {sorted(df['mask_ratio'].unique())}")
    print(f"Reconstruction weights: {sorted(df['reconstruction_weight'].unique())}")
    
    return df

def create_visualizations(df, output_dir):
    """Create visualization plots for the results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Loss comparison by mask ratio
    if 'total_loss' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Total loss by mask ratio
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df, x='mask_ratio', y='total_loss')
        plt.title('Total Loss by Mask Ratio')
        plt.xticks(rotation=45)
        
        # Subplot 2: Recommendation loss by mask ratio
        plt.subplot(2, 2, 2)
        if 'recommendation_loss' in df.columns:
            sns.boxplot(data=df, x='mask_ratio', y='recommendation_loss')
            plt.title('Recommendation Loss by Mask Ratio')
            plt.xticks(rotation=45)
        
        # Subplot 3: Reconstruction loss by mask ratio
        plt.subplot(2, 2, 3)
        if 'reconstruction_loss' in df.columns:
            sns.boxplot(data=df, x='mask_ratio', y='reconstruction_loss')
            plt.title('Reconstruction Loss by Mask Ratio')
            plt.xticks(rotation=45)
        
        # Subplot 4: Heatmap of total loss
        plt.subplot(2, 2, 4)
        pivot_table = df.pivot_table(values='total_loss', 
                                   index='mask_ratio', 
                                   columns='reconstruction_weight', 
                                   aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis')
        plt.title('Total Loss Heatmap')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Parameter sensitivity analysis
    if len(df) > 1:
        plt.figure(figsize=(15, 10))
        
        # Correlation matrix
        numeric_cols = df.select_dtypes(include=[float]).columns
        if len(numeric_cols) > 1:
            plt.subplot(2, 3, 1)
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Parameter Correlation Matrix')
        
        # Scatter plots
        if 'mask_ratio' in df.columns and 'total_loss' in df.columns:
            plt.subplot(2, 3, 2)
            sns.scatterplot(data=df, x='mask_ratio', y='total_loss', 
                          hue='reconstruction_weight', size='reconstruction_weight')
            plt.title('Mask Ratio vs Total Loss')
        
        if 'reconstruction_weight' in df.columns and 'total_loss' in df.columns:
            plt.subplot(2, 3, 3)
            sns.scatterplot(data=df, x='reconstruction_weight', y='total_loss', 
                          hue='mask_ratio', size='mask_ratio')
            plt.title('Reconstruction Weight vs Total Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📈 Visualizations saved to {output_dir}")

def generate_report(df, output_dir):
    """Generate a comprehensive text report."""
    report_file = os.path.join(output_dir, 'experiment_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("MHL Model Masking Experiment Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Experiment Summary:\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Categories: {', '.join(df['category'].unique())}\n")
        f.write(f"Mask ratios tested: {sorted(df['mask_ratio'].unique())}\n")
        f.write(f"Reconstruction weights tested: {sorted(df['reconstruction_weight'].unique())}\n\n")
        
        if 'total_loss' in df.columns:
            f.write("Performance Analysis:\n")
            f.write(f"Best total loss: {df['total_loss'].min():.4f}\n")
            f.write(f"Worst total loss: {df['total_loss'].max():.4f}\n")
            f.write(f"Average total loss: {df['total_loss'].mean():.4f}\n\n")
            
            # Best performing configuration
            best_idx = df['total_loss'].idxmin()
            best_config = df.loc[best_idx]
            f.write("Best Configuration:\n")
            f.write(f"Mask ratio: {best_config['mask_ratio']}\n")
            f.write(f"Reconstruction weight: {best_config['reconstruction_weight']}\n")
            f.write(f"Total loss: {best_config['total_loss']:.4f}\n")
            if 'recommendation_loss' in best_config:
                f.write(f"Recommendation loss: {best_config['recommendation_loss']:.4f}\n")
            if 'reconstruction_loss' in best_config:
                f.write(f"Reconstruction loss: {best_config['reconstruction_loss']:.4f}\n")
    
    print(f"📄 Report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MHL masking experiment results')
    parser.add_argument('--results_dir', type=str, default='results/masking_experiments',
                       help='Directory containing experiment log files')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print("🔍 Analyzing MHL masking experiment results...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Analyze results
    df = analyze_results(args.results_dir)
    
    if df is not None:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save raw data
        df.to_csv(os.path.join(args.output_dir, 'experiment_results.csv'), index=False)
        print(f"📊 Raw data saved to {args.output_dir}/experiment_results.csv")
        
        # Create visualizations
        create_visualizations(df, args.output_dir)
        
        # Generate report
        generate_report(df, args.output_dir)
        
        print(f"\n✅ Analysis completed! Results saved to {args.output_dir}")
    else:
        print("❌ No results to analyze")

if __name__ == "__main__":
    main()
