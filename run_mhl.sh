#!/bin/bash

# MHL Model Training and Evaluation Script
# Usage: ./run_mhl.sh [category] [gpu_id]
# Example: ./run_mhl.sh Beauty 0

# Default parameters
CATEGORY=${1:-"Beauty"}
GPU_ID=${2:-"0"}

echo "=========================================="
echo "Running MHL Model"
echo "=========================================="
echo "Category: $CATEGORY"
echo "GPU ID: $GPU_ID"
echo "=========================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/mhl_${CATEGORY}_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo "Starting training..."

# Run the model with different configurations based on category
case $CATEGORY in
    "Beauty")
        echo "Running Beauty dataset with optimized parameters..."
        python main.py \
            --model=MHL \
            --category=Beauty \
            --lr=0.01 \
            --temperature=0.03 \
            --n_codebook=32 \
            --num_beams=20 \
            --n_edges=200 \
            --propagation_steps=3 \
            --train_batch_size=32 \
            --eval_batch_size=64 \
            --max_epochs=50 \
            --patience=10 \
            --rand_seed=42 \
            --reproducibility=True \
            2>&1 | tee $LOG_FILE
        ;;
    "Sports_and_Outdoors")
        echo "Running Sports_and_Outdoors dataset with optimized parameters..."
        python main.py \
            --model=MHL \
            --category=Sports_and_Outdoors \
            --lr=0.003 \
            --temperature=0.03 \
            --n_codebook=16 \
            --num_beams=100 \
            --n_edges=30 \
            --propagation_steps=5 \
            --train_batch_size=32 \
            --eval_batch_size=64 \
            --max_epochs=50 \
            --patience=10 \
            --rand_seed=42 \
            --reproducibility=True \
            2>&1 | tee $LOG_FILE
        ;;
    "Toys_and_Games")
        echo "Running Toys_and_Games dataset with optimized parameters..."
        python main.py \
            --model=MHL \
            --category=Toys_and_Games \
            --lr=0.003 \
            --temperature=0.03 \
            --n_codebook=16 \
            --num_beams=200 \
            --n_edges=20 \
            --propagation_steps=3 \
            --train_batch_size=32 \
            --eval_batch_size=64 \
            --max_epochs=50 \
            --patience=10 \
            --rand_seed=42 \
            --reproducibility=True \
            2>&1 | tee $LOG_FILE
        ;;
    "CDs_and_Vinyl")
        echo "Running CDs_and_Vinyl dataset with optimized parameters..."
        python main.py \
            --model=MHL \
            --category=CDs_and_Vinyl \
            --lr=0.001 \
            --temperature=0.03 \
            --n_codebook=64 \
            --num_beams=20 \
            --n_edges=500 \
            --propagation_steps=5 \
            --train_batch_size=32 \
            --eval_batch_size=64 \
            --max_epochs=50 \
            --patience=10 \
            --rand_seed=42 \
            --reproducibility=True \
            2>&1 | tee $LOG_FILE
        ;;
    *)
        echo "Unknown category: $CATEGORY"
        echo "Available categories: Beauty, Sports_and_Outdoors, Toys_and_Games, CDs_and_Vinyl"
        echo "Using default Beauty configuration..."
        python main.py \
            --model=MHL \
            --category=Beauty \
            --lr=0.01 \
            --temperature=0.03 \
            --n_codebook=32 \
            --num_beams=20 \
            --n_edges=200 \
            --propagation_steps=3 \
            --train_batch_size=32 \
            --eval_batch_size=64 \
            --max_epochs=50 \
            --patience=10 \
            --rand_seed=42 \
            --reproducibility=True \
            2>&1 | tee $LOG_FILE
        ;;
esac

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Log file: $LOG_FILE"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed! Check the log file: $LOG_FILE"
    echo "=========================================="
    exit 1
fi
