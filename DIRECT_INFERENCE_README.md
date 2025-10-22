# Direct Embedding Matching Inference

## Overview

This inference mode implements a two-stage approach:
1. **Stage 1**: Generate item codes (32 tokens) using RPG model
2. **Stage 2**: Match generated codes with item corpus using embedding similarity

## How It Works

```python
# Stage 1: Generate codes
generated_tokens = model.generate_codes(user_history)  # (batch_size, 32)

# Stage 2: Match with corpus
generated_embeddings = embedding_layer(generated_tokens).mean(dim=1)  # (batch_size, n_embd)
item_embeddings = embedding_layer(all_item_tokens).mean(dim=1)  # (n_items, n_embd)
similarities = cosine_similarity(generated_embeddings, item_embeddings)  # (batch_size, n_items)
top_k_items = similarities.topk(k)
```

## Hyperparameters

### 1. `normalize_embeddings` (default: `True`)
- **Type**: Boolean
- **Description**: Whether to L2-normalize embeddings before computing similarity
- **Effect**:
  - `True`: Compute cosine similarity (recommended)
  - `False`: Compute dot product similarity

**Example**:
```bash
python infer.py --normalize_embeddings=True
```

### 2. `similarity_temperature` (default: `1.0`)
- **Type**: Float
- **Description**: Temperature parameter for similarity scores
- **Effect**:
  - Lower values (e.g., `0.1`): More peaked distribution, confident predictions
  - Higher values (e.g., `10.0`): More uniform distribution, diverse predictions
  - `1.0`: No scaling

**Example**:
```bash
python infer.py --similarity_temperature=0.5
```

## Usage Examples

### Basic Usage (Default Settings)
```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model=RPG \
    --checkpoint=ckpt/RPG_Beauty.pt \
    --infer_mode=direct \
    --category=Beauty
```

### With Custom Hyperparameters
```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model=RPG \
    --checkpoint=ckpt/RPG_Beauty.pt \
    --infer_mode=direct \
    --category=Beauty \
    --normalize_embeddings=True \
    --similarity_temperature=0.5
```

### Test Different Temperatures
```bash
# More confident predictions
python infer.py --infer_mode=direct --similarity_temperature=0.1

# More diverse predictions  
python infer.py --infer_mode=direct --similarity_temperature=2.0
```

## Configuration File

You can also set these parameters in `genrec/models/RPG/config.yaml`:

```yaml
# Direct mode configs (embedding similarity)
normalize_embeddings: True
similarity_temperature: 1.0
```

## Comparison with Other Modes

### Direct Mode (This)
- **Pros**: 
  - Fast inference
  - No graph construction needed
  - Works well for dense item spaces
- **Cons**: 
  - May not capture complex item relationships

### Graph Mode
- **Pros**: 
  - Captures item-item relationships
  - More accurate for sparse data
- **Cons**: 
  - Requires graph construction
  - Slower inference

### Overlap Mode  
- **Pros**:
  - Simple and interpretable
  - Fast
- **Cons**:
  - Requires exact token match
  - Less robust to noise

## Tips

1. **Start with default settings**: `normalize_embeddings=True`, `similarity_temperature=1.0`
2. **For more confident predictions**: Lower `similarity_temperature` to 0.1-0.5
3. **For more diverse predictions**: Raise `similarity_temperature` to 2.0-10.0
4. **Always use normalization**: Unless you have a specific reason not to

## Troubleshooting

**Q: Results are too concentrated on few items?**
A: Increase `similarity_temperature` to make predictions more diverse.

**Q: Results are too random?**
A: Decrease `similarity_temperature` to make predictions more confident.

**Q: Want to use dot product instead of cosine similarity?**
A: Set `normalize_embeddings=False`.

