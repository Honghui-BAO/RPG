#!/bin/bash

# Inference script for RPG model
# Usage: bash run_infer.sh <checkpoint_path> [options]

CHECKPOINT=${1:-"ckpt/RPG.pth"}
CATEGORY=${2:-"Sports_and_Outdoors"}
USE_GRAPH=${3:-""}  # Leave empty for direct embedding matching, set to "--use_graph" for graph-based

echo "Running inference..."
echo "Checkpoint: $CHECKPOINT"
echo "Category: $CATEGORY"

if [ -z "$USE_GRAPH" ]; then
    echo "Mode: Direct Embedding Matching"
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --checkpoint=$CHECKPOINT \
        --category=$CATEGORY \
        --split=test
else
    echo "Mode: Graph-Constrained Decoding"
    CUDA_VISIBLE_DEVICES=0 python infer.py \
        --checkpoint=$CHECKPOINT \
        --category=$CATEGORY \
        --split=test \
        --use_graph
fi

