# RPG Inference Modes Summary

## Three Inference Modes

### 1. Direct Embedding Matching (`direct`)
**Two-stage approach with embedding similarity**

```
Stage 1: Generate codes
user_history → RPG → generated_codes (32 tokens)

Stage 2: Match with corpus  
generated_embeddings ← average(token_embeddings(generated_codes))
item_embeddings ← average(token_embeddings(all_item_codes))
similarity ← cosine_similarity(generated_embeddings, item_embeddings)
top_k ← similarity.topk(k)
```

**Hyperparameters**:
- `normalize_embeddings`: Whether to normalize (default: `True`)
- `similarity_temperature`: Temperature for scores (default: `1.0`)

**When to use**: Fast inference, dense item spaces

---

### 2. Graph-Constrained Decoding (`graph`)
**Original RPG method with item-item graph**

```
Stage 1: Build item-item similarity graph (offline)
Stage 2: Graph propagation
  - Start from random nodes
  - Propagate through edges
  - Select top-k by token logits
```

**Hyperparameters**:
- `num_beams`: Number of beam search paths (default: `50`)
- `n_edges`: Number of edges per node (default: `50`)
- `propagation_steps`: Number of propagation steps (default: `3`)

**When to use**: Best accuracy, sparse data

---

### 3. Token Overlap Counting (`overlap`)
**Two-stage approach with token matching**

```
Stage 1: Generate codes
user_history → RPG → generated_codes (32 tokens)

Stage 2: Count overlaps
for each item in corpus:
    overlap_count = sum(generated_codes == item_codes)
top_k ← overlap_count.topk(k)
```

**Hyperparameters**: None

**When to use**: Simple baseline, interpretable results

---

## Quick Comparison

| Mode | Speed | Accuracy | Memory | Setup |
|------|-------|----------|--------|-------|
| **direct** | ⚡⚡⚡ Fast | 😊 Good | 💾 Low | ✅ None |
| **graph** | 🐢 Slow | 🎯 Best | 💾💾 High | 📊 Build graph |
| **overlap** | ⚡⚡⚡ Fast | 😐 OK | 💾 Low | ✅ None |

---

## Usage Examples

### Direct Mode (Recommended for most cases)
```bash
python infer.py \
    --checkpoint=ckpt/RPG_Beauty.pt \
    --infer_mode=direct \
    --normalize_embeddings=True \
    --similarity_temperature=1.0
```

### Graph Mode (Best accuracy)
```bash
python infer.py \
    --checkpoint=ckpt/RPG_Beauty.pt \
    --infer_mode=graph \
    --num_beams=50 \
    --n_edges=50 \
    --propagation_steps=3
```

### Overlap Mode (Simple baseline)
```bash
python infer.py \
    --checkpoint=ckpt/RPG_Beauty.pt \
    --infer_mode=overlap
```

---

## Hyperparameter Tuning Guide

### Direct Mode
1. **Start**: `normalize_embeddings=True`, `similarity_temperature=1.0`
2. **More confident**: `similarity_temperature=0.1~0.5`
3. **More diverse**: `similarity_temperature=2.0~10.0`

### Graph Mode
1. **Faster**: Reduce `propagation_steps` (1-2)
2. **Better quality**: Increase `num_beams` (100-200)
3. **Denser graph**: Increase `n_edges` (100-200)

---

## Implementation Details

### Two-Stage Generation Process

All three modes follow a similar pattern:

1. **Generate codes**: RPG predicts 32 tokens (semantic codes)
2. **Match with corpus**: Different strategies to find similar items

```python
# Stage 1: Generate codes (shared by all modes)
logits = model.forward(user_history)
generated_tokens = logits.argmax(dim=-1)  # (batch_size, 32)

# Stage 2: Match (different for each mode)
if mode == 'direct':
    # Use embedding similarity
    preds = embedding_match(generated_tokens, all_items)
elif mode == 'graph':
    # Use graph propagation
    preds = graph_propagation(token_logits, graph)
elif mode == 'overlap':
    # Count token overlaps
    preds = overlap_count(generated_tokens, all_items)
```

This design allows you to:
- ✅ See what codes are generated
- ✅ Control the matching strategy  
- ✅ Easily add new matching methods

