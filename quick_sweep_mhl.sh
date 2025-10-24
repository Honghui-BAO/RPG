#!/bin/bash

# Quick Parameter Sweep for MHL Model
# Usage: ./quick_sweep_mhl.sh [category] [gpu_id]

echo "🔬 Quick MHL Parameter Sweep"
echo "=============================="

# Default parameters
CATEGORY=${1:-"Beauty"}
GPU_ID=${2:-"0"}

echo "📊 Dataset: $CATEGORY"
echo "🖥️  GPU: $GPU_ID"
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create results directory
mkdir -p results/quick_sweep
RESULTS_DIR="results/quick_sweep"

# Define parameter ranges
MASK_RATIOS=(0.0 0.1 0.15 0.2 0.25)
RECONSTRUCTION_WEIGHTS=(0.0 0.3 0.5 0.7 1.0)

echo "🧪 Testing parameter combinations:"
echo "Mask ratios: ${MASK_RATIOS[@]}"
echo "Reconstruction weights: ${RECONSTRUCTION_WEIGHTS[@]}"
echo ""

# Counter
exp_count=0
total_experiments=$((${#MASK_RATIOS[@]} * ${#RECONSTRUCTION_WEIGHTS[@]}))

echo "Total experiments: $total_experiments"
echo ""

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
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_file="${RESULTS_DIR}/${CATEGORY}_${exp_name}_${timestamp}.log"
        
        # Run experiment
        python main.py \
            --model=MHL \
            --category=$CATEGORY \
            --gpu=$GPU_ID \
            --mask_ratio=$mask_ratio \
            --reconstruction_weight=$recon_weight \
            --max_epochs=5 \
            --train_batch_size=16 \
            --eval_batch_size=32 \
            --rand_seed=42 \
            --reproducibility=True \
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

echo "🎉 Quick parameter sweep completed!"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "📊 Summary:"
echo "Total experiments: $total_experiments"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "To analyze results, check the log files in the results directory."
