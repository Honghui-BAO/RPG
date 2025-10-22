#!/bin/bash
# Test script for embedding similarity inference

# Change to project root directory
cd "$(dirname "$0")" || exit 1

# Set proxy if needed
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Test embedding similarity inference
echo "Testing embedding similarity inference..."
echo "=========================================="

# You need to provide the checkpoint path
CHECKPOINT_PATH="path/to/your/checkpoint.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    echo "Please update the CHECKPOINT_PATH variable in this script"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python test_embedding_similarity.py \
    --model=RPG \
    --dataset=AmazonReviews2014 \
    --checkpoint="$CHECKPOINT_PATH" \
    --category=Beauty \
    --split=test \
    --batch_size=32 \
    --topk=10

echo "=========================================="
echo "Embedding similarity test completed!"
