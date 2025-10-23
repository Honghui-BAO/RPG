#!/bin/bash

# Quick Test Script for MHL Model with Token Masking
# Usage: ./quick_test_masking.sh [category] [gpu_id]

echo "🧪 Quick Test: MHL Model with Token Masking"
echo "=========================================="

# Default parameters
CATEGORY=${1:-"Beauty"}
GPU_ID=${2:-"0"}

echo "📊 Dataset: $CATEGORY"
echo "🖥️  GPU: $GPU_ID"
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Test different masking configurations
echo "🔬 Testing different masking configurations..."
echo ""

# Test 1: No masking
echo "Test 1: No masking (baseline)"
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
    --max_epochs=3 \
    --patience=2 \
    --mask_ratio=0.0 \
    --reconstruction_weight=0.0

echo ""
echo "Test 2: Light masking (10%)"
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
    --max_epochs=3 \
    --patience=2 \
    --mask_ratio=0.1 \
    --reconstruction_weight=0.3

echo ""
echo "Test 3: Medium masking (15%)"
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
    --max_epochs=3 \
    --patience=2 \
    --mask_ratio=0.15 \
    --reconstruction_weight=0.5

echo ""
echo "Test 4: Heavy masking (25%)"
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
    --max_epochs=3 \
    --patience=2 \
    --mask_ratio=0.25 \
    --reconstruction_weight=0.7

echo ""
echo "✅ All masking tests completed!"
echo "Check the logs to compare performance across different masking configurations."
