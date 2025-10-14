#!/bin/bash
# Training script for LLaDA model

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model=LLADA \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=8 \
    --diffusion_steps=8 \
    --mask_schedule=linear \
    --codes_per_step=4 \
    --eval_batch_size=128 \
    --epochs=150

