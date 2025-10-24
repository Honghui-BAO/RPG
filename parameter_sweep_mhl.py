#!/usr/bin/env python3
"""
MHL Model Parameter Sweep Script
扫描MASK_RATIO和RECONSTRUCTION_WEIGHT参数组合
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
import argparse
import itertools
import pandas as pd

def run_experiment(category, gpu_id, mask_ratio, reconstruction_weight, 
                   epochs=10, batch_size=32, output_dir="results/parameter_sweep"):
    """运行单个实验"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成实验名称
    exp_name = f"mask_{mask_ratio}_recon_{reconstruction_weight}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{category}_{exp_name}_{timestamp}.log")
    
    print(f"🧪 Running experiment: {exp_name}")
    print(f"   Category: {category}")
    print(f"   GPU: {gpu_id}")
    print(f"   Mask ratio: {mask_ratio}")
    print(f"   Reconstruction weight: {reconstruction_weight}")
    print(f"   Log file: {log_file}")
    
    # 构建命令
    cmd = [
        'python', 'main.py',
        '--model=MHL',
        f'--category={category}',
        f'--gpu={gpu_id}',
        f'--mask_ratio={mask_ratio}',
        f'--reconstruction_weight={reconstruction_weight}',
        f'--max_epochs={epochs}',
        f'--train_batch_size={batch_size}',
        f'--eval_batch_size={batch_size}',
        '--rand_seed=42',
        '--reproducibility=True'
    ]
    
    # 运行实验
    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            f.write(f"MHL Parameter Sweep Experiment\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 50 + "\n\n")
            
            # 运行命令
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=3600  # 1小时超时
            )
            
            # 写入日志
            f.write(result.stdout)
            f.write(f"\n\nExit code: {result.returncode}\n")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 解析结果
        success = result.returncode == 0
        if success:
            print(f"✅ Experiment completed successfully in {duration:.1f}s")
        else:
            print(f"❌ Experiment failed after {duration:.1f}s")
        
        return {
            'exp_name': exp_name,
            'category': category,
            'gpu_id': gpu_id,
            'mask_ratio': mask_ratio,
            'reconstruction_weight': reconstruction_weight,
            'epochs': epochs,
            'batch_size': batch_size,
            'success': success,
            'duration': duration,
            'log_file': log_file,
            'exit_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out after 1 hour")
        return {
            'exp_name': exp_name,
            'category': category,
            'gpu_id': gpu_id,
            'mask_ratio': mask_ratio,
            'reconstruction_weight': reconstruction_weight,
            'epochs': epochs,
            'batch_size': batch_size,
            'success': False,
            'duration': 3600,
            'log_file': log_file,
            'exit_code': -1
        }
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        return {
            'exp_name': exp_name,
            'category': category,
            'gpu_id': gpu_id,
            'mask_ratio': mask_ratio,
            'reconstruction_weight': reconstruction_weight,
            'epochs': epochs,
            'batch_size': batch_size,
            'success': False,
            'duration': 0,
            'log_file': log_file,
            'exit_code': -2
        }

def parse_results(log_file):
    """解析实验结果"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 提取关键指标
        results = {}
        
        # 提取最终损失
        if 'Test Results:' in content:
            # 这里需要根据实际日志格式调整
            pass
        
        return results
    except Exception as e:
        print(f"Warning: Could not parse results from {log_file}: {e}")
        return {}

def run_parameter_sweep(category, gpu_id, mask_ratios, reconstruction_weights, 
                       epochs=10, batch_size=32, output_dir="results/parameter_sweep"):
    """运行参数扫描"""
    
    print("🔬 MHL Model Parameter Sweep")
    print("=" * 50)
    print(f"Category: {category}")
    print(f"GPU: {gpu_id}")
    print(f"Mask ratios: {mask_ratios}")
    print(f"Reconstruction weights: {reconstruction_weights}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成所有参数组合
    param_combinations = list(itertools.product(mask_ratios, reconstruction_weights))
    total_experiments = len(param_combinations)
    
    print(f"Total experiments: {total_experiments}")
    print()
    
    # 运行实验
    results = []
    successful_experiments = 0
    
    for i, (mask_ratio, reconstruction_weight) in enumerate(param_combinations, 1):
        print(f"📊 Experiment {i}/{total_experiments}")
        
        result = run_experiment(
            category=category,
            gpu_id=gpu_id,
            mask_ratio=mask_ratio,
            reconstruction_weight=reconstruction_weight,
            epochs=epochs,
            batch_size=batch_size,
            output_dir=output_dir
        )
        
        results.append(result)
        
        if result['success']:
            successful_experiments += 1
        
        print(f"   Success: {result['success']}")
        print(f"   Duration: {result['duration']:.1f}s")
        print()
    
    # 保存结果
    results_file = os.path.join(output_dir, f"{category}_parameter_sweep_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 创建结果摘要
    summary_file = os.path.join(output_dir, f"{category}_parameter_sweep_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("MHL Model Parameter Sweep Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Successful experiments: {successful_experiments}\n")
        f.write(f"Success rate: {successful_experiments/total_experiments*100:.1f}%\n\n")
        
        f.write("Parameter combinations tested:\n")
        for mask_ratio in mask_ratios:
            for recon_weight in reconstruction_weights:
                f.write(f"  Mask ratio: {mask_ratio}, Reconstruction weight: {recon_weight}\n")
        
        f.write("\nDetailed results:\n")
        for result in results:
            f.write(f"  {result['exp_name']}: {'✅' if result['success'] else '❌'} "
                   f"({result['duration']:.1f}s)\n")
    
    # 创建CSV文件
    df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, f"{category}_parameter_sweep_results.csv")
    df.to_csv(csv_file, index=False)
    
    print("🎉 Parameter sweep completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📄 Summary: {summary_file}")
    print(f"📊 CSV: {csv_file}")
    print(f"📋 JSON: {results_file}")
    print()
    print(f"✅ Successful experiments: {successful_experiments}/{total_experiments}")
    print(f"📈 Success rate: {successful_experiments/total_experiments*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='MHL Model Parameter Sweep')
    parser.add_argument('--category', type=str, default='Beauty',
                       choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'CDs_and_Vinyl'],
                       help='Dataset category')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--mask_ratios', type=float, nargs='+', 
                       default=[0.0, 0.1, 0.15, 0.2, 0.25],
                       help='Mask ratios to test')
    parser.add_argument('--reconstruction_weights', type=float, nargs='+',
                       default=[0.0, 0.3, 0.5, 0.7, 1.0],
                       help='Reconstruction weights to test')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs for each experiment')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='results/parameter_sweep',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 运行参数扫描
    run_parameter_sweep(
        category=args.category,
        gpu_id=args.gpu,
        mask_ratios=args.mask_ratios,
        reconstruction_weights=args.reconstruction_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
