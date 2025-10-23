#!/bin/bash
# Training script for MHL (Masked Hierarchical Learning) model

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Set proxy (if needed)
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# MHL Training Configuration
echo "Starting MHL (Masked Hierarchical Learning) model training..."
echo "Model: MHL"
echo "Dataset: Beauty"
echo "Mask Ratio: 0.15 (15% of tokens masked during training)"
echo "Dual Loss: Reconstruction + Next-item Prediction"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model=MHL \
    --category=Beauty \
    --run_id=mhl_beauty \
    --lr=0.01 \
    --temperature=0.07 \
    --n_codebook=32 \
    --mask_ratio=0.15 \
    --recon_loss_weight=1.0 \
    --next_item_loss_weight=1.0 \
    --num_beams=20 \
    --n_edges=200 \
    --propagation_steps=3 \
    --epochs=100 \
    --batch_size=128 \
    --eval_batch_size=128 \
    --save_steps=1000 \
    --eval_steps=500 \
    --logging_steps=100 \
    --warmup_steps=1000 \
    --max_grad_norm=1.0 \
    --weight_decay=0.01

echo "MHL training completed!"
