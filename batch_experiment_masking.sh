#!/bin/bash

# Batch Experiment Script for MHL Model with Token Masking
# Tests different combinations of masking parameters
# Usage: ./batch_experiment_masking.sh [category] [gpu_id]

echo "🔬 Batch Experiment: MHL Model with Token Masking"
echo "=============================================="

# Default parameters
CATEGORY=${1:-"Beauty"}
GPU_ID=${2:-"0"}

echo "📊 Dataset: $CATEGORY"
echo "🖥️  GPU: $GPU_ID"
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create results directory
mkdir -p results/masking_experiments
RESULTS_DIR="results/masking_experiments"

# Define parameter combinations
MASK_RATIOS=(0.0 0.1 0.15 0.2 0.25)
RECONSTRUCTION_WEIGHTS=(0.0 0.3 0.5 0.7)

echo "🧪 Running batch experiments..."
echo "Mask ratios: ${MASK_RATIOS[@]}"
echo "Reconstruction weights: ${RECONSTRUCTION_WEIGHTS[@]}"
echo ""

# Counter for experiments
exp_count=0
total_experiments=$((${#MASK_RATIOS[@]} * ${#RECONSTRUCTION_WEIGHTS[@]}))

# Run experiments
for mask_ratio in "${MASK_RATIOS[@]}"; do
    for recon_weight in "${RECONSTRUCTION_WEIGHTS[@]}"; do
        exp_count=$((exp_count + 1))
        
        echo "=========================================="
        echo "Experiment $exp_count/$total_experiments"
        echo "Mask ratio: $mask_ratio"
        echo "Reconstruction weight: $recon_weight"
        echo "=========================================="
        
        # Create experiment name
        exp_name="mask_${mask_ratio}_recon_${recon_weight}"
        log_file="${RESULTS_DIR}/${CATEGORY}_${exp_name}_$(date +%Y%m%d_%H%M%S).log"
        
        # Run experiment
        python main.py \
            --model=MHL \
            --category=$CATEGORY \
            --lr=0.01 \
            --temperature=0.03 \
            --n_codebook=32 \
            --num_beams=20 \
            --n_edges=200 \
            --propagation_steps=3 \
            --train_batch_size=32 \
            --eval_batch_size=64 \
            --max_epochs=20 \
            --patience=5 \
            --rand_seed=42 \
            --reproducibility=True \
            --mask_ratio=$mask_ratio \
            --reconstruction_weight=$recon_weight \
            2>&1 | tee $log_file
        
        # Check if experiment completed successfully
        if [ $? -eq 0 ]; then
            echo "✅ Experiment $exp_count completed successfully"
        else
            echo "❌ Experiment $exp_count failed"
        fi
        
        echo ""
    done
done

echo "🎉 All experiments completed!"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "📊 Summary of experiments:"
echo "Total experiments: $total_experiments"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "To analyze results, check the log files in the results directory."
