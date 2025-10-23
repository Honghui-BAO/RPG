#!/bin/bash
# Testing script for MHL (Masked Hierarchical Learning) model

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Set proxy (if needed)
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# MHL Testing Configuration
echo "Starting MHL (Masked Hierarchical Learning) model testing..."
echo "Model: MHL"
echo "Dataset: Beauty"
echo "Mode: Evaluation (no masking during inference)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model=MHL \
    --category=Beauty \
    --checkpoint=./checkpoints/MHL/Beauty/best_model.pt \
    --eval_only=true \
    --eval_batch_size=256 \
    --num_beams=20 \
    --n_edges=200 \
    --propagation_steps=3 \
    --temperature=0.07

echo "MHL testing completed!"
