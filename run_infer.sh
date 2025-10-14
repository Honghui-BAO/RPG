#!/bin/bash

# Inference script for RPG model
# Usage: bash run_infer.sh <checkpoint_path> <category> [infer_mode]
# infer_mode: direct (default), graph, overlap

CHECKPOINT=${1:-"ckpt/RPG.pth"}
CATEGORY=${2:-"Sports_and_Outdoors"}
INFER_MODE=${3:-"direct"}  # direct, graph, or overlap

echo "Running inference..."
echo "Checkpoint: $CHECKPOINT"
echo "Category: $CATEGORY"
echo "Inference Mode: $INFER_MODE"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --checkpoint=$CHECKPOINT \
    --category=$CATEGORY \
    --split=test \
    --infer_mode=$INFER_MODE

