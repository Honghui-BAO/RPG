#!/bin/bash
# Fast training script for MHL (Masked Hierarchical Learning) model
# Quick test with fewer epochs for development and debugging

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Set proxy (if needed)
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# MHL Fast Training Configuration
echo "Starting MHL (Masked Hierarchical Learning) FAST training..."
echo "Model: MHL"
echo "Dataset: Beauty"
echo "Mode: Fast training (fewer epochs for testing)"
echo "Mask Ratio: 0.15"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model=MHL \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.07 \
    --n_codebook=32 \
    --mask_ratio=0.15 \
    --recon_loss_weight=1.0 \
    --next_item_loss_weight=1.0 \
    --epochs=10 \
    --batch_size=128 \
    --eval_batch_size=128 \
    --save_steps=100 \
    --eval_steps=50 \
    --logging_steps=10 \
    --warmup_steps=100 \
    --max_grad_norm=1.0 \
    --weight_decay=0.01

echo "MHL fast training completed!"
