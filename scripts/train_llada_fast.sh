#!/bin/bash
# Fast training for LLADA with aggressive inference speedup

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Aggressive speedup: 16 codes per step = only 2 steps!
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model=LLADA \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --diffusion_steps=32 \
    --mask_schedule=linear \
    --codes_per_step=16 \
    --eval_batch_size=256 \
    --epochs=150

