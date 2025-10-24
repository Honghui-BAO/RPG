#!/usr/bin/env python3
"""
分析参数扫描结果的脚本
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import argparse
import glob

def parse_log_file(log_file):
    """解析单个日志文件，提取关键指标"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 提取参数
        mask_ratio = None
        reconstruction_weight = None
        
        # 从文件名提取参数
        filename = os.path.basename(log_file)
        if 'mask_' in filename and 'recon_' in filename:
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if part == 'mask' and i + 1 < len(parts):
                    mask_ratio = float(parts[i + 1])
                elif part == 'recon' and i + 1 < len(parts):
                    reconstruction_weight = float(parts[i + 1].split('.')[0])
        
        # 提取训练指标
        metrics = {
            'mask_ratio': mask_ratio,
            'reconstruction_weight': reconstruction_weight,
            'log_file': log_file,
            'success': True
        }
        
        # 提取最终损失（需要根据实际日志格式调整）
        if 'Test Results:' in content:
            # 这里需要根据实际日志格式提取指标
            pass
        
        # 提取训练时间
        if 'Training completed' in content:
            metrics['training_completed'] = True
        else:
            metrics['training_completed'] = False
        
        return metrics
        
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None

def analyze_results(results_dir):
    """分析参数扫描结果"""
    print("🔍 Analyzing parameter sweep results...")
    
    # 查找所有日志文件
    log_files = glob.glob(os.path.join(results_dir, "*.log"))
    
    if not log_files:
        print(f"No log files found in {results_dir}")
        return None
    
    print(f"Found {len(log_files)} log files")
    
    # 解析所有日志文件
    results = []
    for log_file in log_files:
        metrics = parse_log_file(log_file)
        if metrics:
            results.append(metrics)
    
    if not results:
        print("No valid results found")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 基本统计
    print("\n📊 Results Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Successful experiments: {df['training_completed'].sum()}")
    print(f"Success rate: {df['training_completed'].mean()*100:.1f}%")
    
    # 参数分布
    print(f"\nParameter ranges:")
    print(f"Mask ratios: {sorted(df['mask_ratio'].unique())}")
    print(f"Reconstruction weights: {sorted(df['reconstruction_weight'].unique())}")
    
    return df

def create_visualizations(df, output_dir):
    """创建可视化图表"""
    print("📈 Creating visualizations...")
    
    # 设置样式
    plt.style.use('seaborn-v0_8')
    
    # 1. 参数组合热力图
    plt.figure(figsize=(12, 8))
    
    # 创建参数组合矩阵
    mask_ratios = sorted(df['mask_ratio'].unique())
    recon_weights = sorted(df['reconstruction_weight'].unique())
    
    # 计算成功率矩阵
    success_matrix = np.zeros((len(mask_ratios), len(recon_weights)))
    for i, mask_ratio in enumerate(mask_ratios):
        for j, recon_weight in enumerate(recon_weights):
            subset = df[(df['mask_ratio'] == mask_ratio) & (df['reconstruction_weight'] == recon_weight)]
            if len(subset) > 0:
                success_matrix[i, j] = subset['training_completed'].mean()
    
    # 绘制热力图
    sns.heatmap(success_matrix, 
                xticklabels=recon_weights, 
                yticklabels=mask_ratios,
                annot=True, 
                fmt='.2f', 
                cmap='RdYlGn',
                cbar_kws={'label': 'Success Rate'})
    
    plt.title('Parameter Sweep Success Rate Heatmap')
    plt.xlabel('Reconstruction Weight')
    plt.ylabel('Mask Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 参数分布图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mask ratio分布
    axes[0].hist(df['mask_ratio'], bins=20, alpha=0.7, color='skyblue')
    axes[0].set_title('Mask Ratio Distribution')
    axes[0].set_xlabel('Mask Ratio')
    axes[0].set_ylabel('Frequency')
    
    # Reconstruction weight分布
    axes[1].hist(df['reconstruction_weight'], bins=20, alpha=0.7, color='lightcoral')
    axes[1].set_title('Reconstruction Weight Distribution')
    axes[1].set_xlabel('Reconstruction Weight')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['reconstruction_weight'], df['mask_ratio'], 
                         c=df['training_completed'], cmap='RdYlGn', s=100, alpha=0.7)
    plt.colorbar(scatter, label='Training Completed')
    plt.xlabel('Reconstruction Weight')
    plt.ylabel('Mask Ratio')
    plt.title('Parameter Combinations Scatter Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}")

def generate_report(df, output_dir):
    """生成分析报告"""
    print("📄 Generating analysis report...")
    
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("MHL Model Parameter Sweep Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Experiment Summary:\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Successful experiments: {df['training_completed'].sum()}\n")
        f.write(f"Success rate: {df['training_completed'].mean()*100:.1f}%\n\n")
        
        f.write("Parameter Analysis:\n")
        f.write(f"Mask ratios tested: {sorted(df['mask_ratio'].unique())}\n")
        f.write(f"Reconstruction weights tested: {sorted(df['reconstruction_weight'].unique())}\n\n")
        
        # 最佳参数组合
        successful_experiments = df[df['training_completed'] == True]
        if len(successful_experiments) > 0:
            f.write("Successful parameter combinations:\n")
            for _, row in successful_experiments.iterrows():
                f.write(f"  Mask ratio: {row['mask_ratio']}, "
                       f"Reconstruction weight: {row['reconstruction_weight']}\n")
        else:
            f.write("No successful experiments found.\n")
        
        f.write("\nRecommendations:\n")
        f.write("1. Check log files for failed experiments to identify issues\n")
        f.write("2. Consider adjusting parameter ranges based on results\n")
        f.write("3. Run longer experiments for successful parameter combinations\n")
    
    print(f"📄 Report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MHL parameter sweep results')
    parser.add_argument('--results_dir', type=str, default='results/quick_sweep',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print("🔍 MHL Parameter Sweep Results Analysis")
    print("=" * 50)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析结果
    df = analyze_results(args.results_dir)
    
    if df is not None:
        # 保存原始数据
        df.to_csv(os.path.join(args.output_dir, 'analysis_data.csv'), index=False)
        print(f"📊 Raw data saved to {args.output_dir}/analysis_data.csv")
        
        # 创建可视化
        create_visualizations(df, args.output_dir)
        
        # 生成报告
        generate_report(df, args.output_dir)
        
        print(f"\n✅ Analysis completed! Results saved to {args.output_dir}")
    else:
        print("❌ No results to analyze")

if __name__ == "__main__":
    main()
