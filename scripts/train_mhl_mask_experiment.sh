#!/bin/bash
# Experiment script for MHL model with different masking ratios
# Tests various mask_ratio values to find optimal masking strategy

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Set proxy (if needed)
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Experiment configuration
echo "Starting MHL Masking Ratio Experiment..."
echo "Testing different mask_ratio values: 0.05, 0.10, 0.15, 0.20, 0.25"
echo "=========================================="

# Array of mask ratios to test
mask_ratios=(0.05 0.10 0.15 0.20 0.25)

for mask_ratio in "${mask_ratios[@]}"; do
    echo "Training MHL with mask_ratio=$mask_ratio..."
    
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --model=MHL \
        --category=Beauty \
        --lr=0.01 \
        --temperature=0.07 \
        --n_codebook=32 \
        --mask_ratio=$mask_ratio \
        --recon_loss_weight=1.0 \
        --next_item_loss_weight=1.0 \
        --epochs=50 \
        --batch_size=128 \
        --eval_batch_size=128 \
        --save_steps=500 \
        --eval_steps=250 \
        --logging_steps=50 \
        --warmup_steps=500 \
        --max_grad_norm=1.0 \
        --weight_decay=0.01 \
        --run_id="MHL_mask_${mask_ratio}"
    
    echo "Completed training with mask_ratio=$mask_ratio"
    echo "----------------------------------------"
done

echo "MHL masking ratio experiment completed!"
echo "Check logs for performance comparison across different mask ratios."
