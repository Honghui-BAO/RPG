# RPG Inference Guide

This branch (`rpg_infer`) provides flexible inference options for the RPG model with different decoding strategies.

## Features

- **Direct Embedding Matching**: Fast inference by computing scores for all items using token embeddings
- **Graph-Constrained Decoding**: Efficient search using pre-computed item similarity graph
- **Item Visit Statistics**: Track the number of items visited during inference

## Usage

### Basic Inference (Direct Embedding Matching)

```bash
# Using the convenience script
bash run_infer.sh <checkpoint_path> <category>

# Example
bash run_infer.sh ckpt/RPG_Sports_and_Outdoors.pth Sports_and_Outdoors
```

### Graph-Based Inference

```bash
# Using the convenience script with graph mode
bash run_infer.sh <checkpoint_path> <category> --use_graph

# Example
bash run_infer.sh ckpt/RPG_Sports_and_Outdoors.pth Sports_and_Outdoors --use_graph
```

### Advanced Usage (Direct Python Call)

```bash
# Direct embedding matching (default)
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --checkpoint=ckpt/RPG.pth \
    --category=Sports_and_Outdoors \
    --split=test

# Graph-constrained decoding
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --checkpoint=ckpt/RPG.pth \
    --category=Sports_and_Outdoors \
    --split=test \
    --use_graph

# Inference on validation set
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --checkpoint=ckpt/RPG.pth \
    --category=Beauty \
    --split=val
```

## Inference Strategies Comparison

### 1. Direct Embedding Matching (Default)
- **Speed**: Very fast (single forward pass)
- **Items Visited**: All items in the catalog
- **Memory**: O(n_items × n_tokens)
- **Best for**: Small to medium catalogs, when you need exhaustive search

### 2. Graph-Constrained Decoding (--use_graph)
- **Speed**: Slower (graph propagation steps)
- **Items Visited**: Only graph neighbors (much fewer)
- **Memory**: O(num_beams × propagation_steps × n_edges)
- **Best for**: Large catalogs, when you want to reduce computation

## Output Metrics

The inference script will output:
- `recall@k`: Recall at different k values (e.g., recall@10, recall@20)
- `ndcg@k`: NDCG at different k values
- `n_visited_items`: Average number of items visited during inference

Example output:
```
TEST Results:
  recall@10: 0.0523
  recall@20: 0.0821
  ndcg@10: 0.0312
  ndcg@20: 0.0398
  n_visited_items: 18357.0000  # Direct matching visits all items
```

## Configuration

You can override any model/dataset configuration via command line:

```bash
python infer.py \
    --checkpoint=ckpt/RPG.pth \
    --category=Beauty \
    --n_codebook=32 \
    --temperature=0.03 \
    --num_beams=100
```

## Code Changes

### Modified Files:
1. `genrec/models/RPG/model.py`:
   - Modified `generate()` to return tuple `(preds, n_visited_items)`
   - Added item visit count for direct embedding matching

2. `genrec/trainer.py`:
   - Updated `evaluate()` to handle tuple returns in non-DDP mode

### New Files:
1. `infer.py`: Standalone inference script
2. `run_infer.sh`: Convenience bash script
3. `INFERENCE_README.md`: This file

## Performance Tips

1. **For Speed**: Use direct embedding matching (default)
2. **For Memory Efficiency**: Use graph-constrained decoding with smaller `num_beams`
3. **For Accuracy**: Compare both methods and choose based on your dataset

## Example Workflow

```bash
# 1. Train model (or use existing checkpoint)
# Already done - model is trained

# 2. Test direct embedding matching
bash run_infer.sh ckpt/model.pth Sports_and_Outdoors

# 3. Test graph-based decoding
bash run_infer.sh ckpt/model.pth Sports_and_Outdoors --use_graph

# 4. Compare results and choose the best strategy
```

## Notes

- The checkpoint must be compatible with the model architecture
- Make sure the dataset is properly downloaded (happens automatically on first run)
- Graph construction (for graph-based mode) happens once and is cached

