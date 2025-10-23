#!/usr/bin/env python3
"""
MHL Model Runner Script
A flexible Python script for running MHL model with different configurations.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def get_default_configs():
    """Get default configurations for different datasets."""
    return {
        'Beauty': {
            'lr': 0.01,
            'temperature': 0.03,
            'n_codebook': 32,
            'num_beams': 20,
            'n_edges': 200,
            'propagation_steps': 3,
            'train_batch_size': 32,
            'eval_batch_size': 64,
            'max_epochs': 50,
            'patience': 10
        },
        'Sports_and_Outdoors': {
            'lr': 0.003,
            'temperature': 0.03,
            'n_codebook': 16,
            'num_beams': 100,
            'n_edges': 30,
            'propagation_steps': 5,
            'train_batch_size': 32,
            'eval_batch_size': 64,
            'max_epochs': 50,
            'patience': 10
        },
        'Toys_and_Games': {
            'lr': 0.003,
            'temperature': 0.03,
            'n_codebook': 16,
            'num_beams': 200,
            'n_edges': 20,
            'propagation_steps': 3,
            'train_batch_size': 32,
            'eval_batch_size': 64,
            'max_epochs': 50,
            'patience': 10
        },
        'CDs_and_Vinyl': {
            'lr': 0.001,
            'temperature': 0.03,
            'n_codebook': 64,
            'num_beams': 20,
            'n_edges': 500,
            'propagation_steps': 5,
            'train_batch_size': 32,
            'eval_batch_size': 64,
            'max_epochs': 50,
            'patience': 10
        }
    }

def build_command(args, config):
    """Build the command to run the model."""
    cmd = [
        'python', 'main.py',
        '--model=MHL',
        f'--category={args.category}',
        f'--lr={config["lr"]}',
        f'--temperature={config["temperature"]}',
        f'--n_codebook={config["n_codebook"]}',
        f'--num_beams={config["num_beams"]}',
        f'--n_edges={config["n_edges"]}',
        f'--propagation_steps={config["propagation_steps"]}',
        f'--train_batch_size={config["train_batch_size"]}',
        f'--eval_batch_size={config["eval_batch_size"]}',
        f'--max_epochs={config["max_epochs"]}',
        f'--patience={config["patience"]}',
        '--rand_seed=42',
        '--reproducibility=True'
    ]
    
    # Add custom parameters if provided
    if args.lr is not None:
        cmd[cmd.index(f'--lr={config["lr"]}')] = f'--lr={args.lr}'
    if args.temperature is not None:
        cmd[cmd.index(f'--temperature={config["temperature"]}')] = f'--temperature={args.temperature}'
    if args.batch_size is not None:
        cmd[cmd.index(f'--train_batch_size={config["train_batch_size"]}')] = f'--train_batch_size={args.batch_size}'
        cmd[cmd.index(f'--eval_batch_size={config["eval_batch_size"]}')] = f'--eval_batch_size={args.batch_size}'
    if args.epochs is not None:
        cmd[cmd.index(f'--max_epochs={config["max_epochs"]}')] = f'--max_epochs={args.epochs}'
    
    return cmd

def run_experiment(args):
    """Run the MHL model experiment."""
    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Get configuration
    configs = get_default_configs()
    if args.category not in configs:
        print(f"❌ Unknown category: {args.category}")
        print(f"Available categories: {', '.join(configs.keys())}")
        return False
    
    config = configs[args.category]
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/mhl_{args.category}_{timestamp}.log"
    
    # Build command
    cmd = build_command(args, config)
    
    print("🚀 MHL Model Runner")
    print("=" * 50)
    print(f"📊 Dataset: {args.category}")
    print(f"🖥️  GPU: {args.gpu}")
    print(f"📝 Log: {log_file}")
    print("=" * 50)
    print("🔧 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 50)
    
    # Run the command
    try:
        with open(log_file, 'w') as f:
            f.write(f"MHL Model Training Log - {datetime.now()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 50 + "\n\n")
            
            # Run the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
            if process.returncode == 0:
                print("\n✅ Training completed successfully!")
                print(f"📝 Log saved to: {log_file}")
                return True
            else:
                print(f"\n❌ Training failed with exit code: {process.returncode}")
                print(f"📝 Check log file: {log_file}")
                return False
                
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='MHL Model Runner')
    parser.add_argument('--category', type=str, default='Beauty',
                       choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'CDs_and_Vinyl'],
                       help='Dataset category to use')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides default)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperature parameter (overrides default)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training and evaluation (overrides default)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Maximum number of epochs (overrides default)')
    
    args = parser.parse_args()
    
    success = run_experiment(args)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
