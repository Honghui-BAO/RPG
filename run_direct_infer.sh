#!/bin/bash
# Inference script using direct embedding matching mode

# Change to project root directory
cd "$(dirname "$0")" || exit 1

# Set proxy if needed
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Configuration
CHECKPOINT_PATH="ckpt/RPG_AmazonReviews2014_Beauty.pt"  # Update this path
CATEGORY="Beauty"
SPLIT="test"
INFER_MODE="direct"

# Hyperparameters for direct embedding matching
NORMALIZE_EMBEDDINGS=True  # True or False
SIMILARITY_TEMPERATURE=1.0  # Lower values make distribution more peaked

echo "=========================================="
echo "Direct Embedding Matching Inference"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Category: $CATEGORY"
echo "Split: $SPLIT"
echo "Inference Mode: $INFER_MODE"
echo ""
echo "Hyperparameters:"
echo "  - normalize_embeddings: $NORMALIZE_EMBEDDINGS"
echo "  - similarity_temperature: $SIMILARITY_TEMPERATURE"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model=RPG \
    --dataset=AmazonReviews2014 \
    --checkpoint="$CHECKPOINT_PATH" \
    --infer_mode="$INFER_MODE" \
    --split="$SPLIT" \
    --category="$CATEGORY" \
    --normalize_embeddings="$NORMALIZE_EMBEDDINGS" \
    --similarity_temperature="$SIMILARITY_TEMPERATURE"

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
