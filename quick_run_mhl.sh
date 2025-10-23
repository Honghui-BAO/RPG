#!/bin/bash

# Quick MHL Model Runner
# Simple script for quick testing

echo "🚀 Quick MHL Model Runner"
echo "=========================="

# Default to Beauty dataset
CATEGORY=${1:-"Beauty"}
GPU=${2:-"0"}

echo "📊 Dataset: $CATEGORY"
echo "🖥️  GPU: $GPU"
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU

# Run with basic parameters
echo "🏃 Starting MHL model training..."
python main.py \
    --model=MHL \
    --category=$CATEGORY \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --num_beams=20 \
    --n_edges=200 \
    --propagation_steps=3 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --max_epochs=10 \
    --patience=5

echo ""
echo "✅ Training completed!"
