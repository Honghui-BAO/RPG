# Token-Level Encoding (llada_token branch)

## Overview

This branch implements **token-level encoding** instead of item-level aggregation before sending to GPT2.

## Key Difference

### Original RPG (item-level aggregation)

```python
# Step 1: Get token embeddings
input_tokens = item_id2tokens[batch['input_ids']]  # (batch, seq_len, 32)
input_embs = gpt2.wte(input_tokens)  # (batch, seq_len, 32, n_embd)

# Step 2: Aggregate at item-level BEFORE GPT2
input_embs = input_embs.mean(dim=-2)  # (batch, seq_len, n_embd)

# Step 3: Send to GPT2
outputs = gpt2(inputs_embeds=input_embs)  # seq_len positions
```

### Token-Level Encoding (this branch)

```python
# Step 1: Get token embeddings
input_tokens = item_id2tokens[batch['input_ids']]  # (batch, seq_len, 32)
input_embs = gpt2.wte(input_tokens)  # (batch, seq_len, 32, n_embd)

# Step 2: NO aggregation - reshape to treat each token as a position
input_embs = input_embs.view(batch, seq_len * 32, n_embd)

# Step 3: Send to GPT2 (seq_len * 32 positions!)
outputs = gpt2(inputs_embeds=input_embs)

# Step 4: Extract item representation from the last token of each item
hidden_states = outputs.view(batch, seq_len, 32, n_embd)
item_repr = hidden_states[:, :, -1, :]  # Take last token of each item
```

## Pros and Cons

### ✅ Advantages

1. **Richer Representation**: 
   - GPT2 can attend to individual tokens
   - Better capture token-level interactions
   - More expressive power

2. **Better Context**:
   - Each token can see other tokens in the sequence
   - Cross-item and cross-token attention

3. **Follows Transformer Philosophy**:
   - Let the model learn what to attend to
   - No premature aggregation

### ⚠️ Potential Concerns

1. **Sequence Length**:
   - Original: `seq_len` positions (e.g., 50)
   - This branch: `seq_len * 32` positions (e.g., 1600)
   - May hit GPT2's max position limit (2048 in config)

2. **Computational Cost**:
   - Attention is O(n²) where n is sequence length
   - 32x longer sequence → ~1000x more attention computation!
   - Memory usage also increases significantly

3. **Parallel Generation Concern**:
   - You mentioned worrying about violating parallel encoding
   - **Actually it's fine!** The tokens are still predicted in parallel
   - The difference is in how we ENCODE the context, not how we GENERATE

## Does This Violate Parallel Generation?

**No, it doesn't!** Here's why:

### Encoding Phase (This branch changes this)
```python
# Context encoding: token-level
context_tokens = [t1, t2, t3, ..., t32] for each item
gpt2_output = gpt2(all_tokens)  # Attend to all tokens
```

### Generation Phase (This stays the same - still parallel!)
```python
# Still predict all 32 tokens in parallel
for i in range(32):
    logits_i = pred_head_i(gpt2_output)  # Predict token i
# All 32 predictions are independent and parallel
```

The key insight: **Encoding can be sequential/token-level, but generation is still parallel!**

## Example

### User history: [item_A, item_B, item_C]

**Original RPG**:
```
GPT2 input: [emb_A, emb_B, emb_C]  # 3 positions
where emb_A = mean([token_1, ..., token_32])
```

**Token-level encoding**:
```
GPT2 input: [t_A1, t_A2, ..., t_A32, t_B1, t_B2, ..., t_B32, t_C1, t_C2, ..., t_C32]
# 96 positions (3 items × 32 tokens)

# Extract item representations:
repr_A = hidden_state[31]   # Last token of item A
repr_B = hidden_state[63]   # Last token of item B  
repr_C = hidden_state[95]   # Last token of item C
```

## Configuration Considerations

### Adjust max_item_seq_len

Since sequence length is multiplied by 32, you may need to reduce `max_item_seq_len`:

```yaml
# Original
max_item_seq_len: 50  # 50 items × 32 tokens = 1600 positions

# Recommended for this branch
max_item_seq_len: 30  # 30 items × 32 tokens = 960 positions (safer)
```

### GPT2 n_positions

Make sure GPT2's `n_positions` is large enough:

```yaml
n_positions: 2048  # Should be > max_item_seq_len * n_codebook
```

## Usage

This branch uses the same interface as the original RPG:

```bash
# Training
python main.py --category=Beauty

# Inference
python infer.py \
    --checkpoint=ckpt/RPG_Beauty.pt \
    --infer_mode=direct \
    --category=Beauty
```

## Performance Expectations

### Training
- **Slower**: ~10-30x slower per batch due to longer sequences
- **More memory**: ~10-30x more GPU memory

### Inference
- **Similar**: Generation is still parallel
- **Slightly slower**: Forward pass through GPT2 takes longer

### Quality
- **Potentially better**: Richer representations
- **Needs experimentation**: May or may not improve final metrics

## Implementation Details

### Attention Mask Expansion

```python
# Original mask: (batch, seq_len)
attention_mask = [1, 1, 1, 0, 0]  # 3 items, 2 padding

# Expanded mask: (batch, seq_len * n_codebook)
attention_mask_expanded = [
    1,1,1,...,1,  # 32 tokens for item 1
    1,1,1,...,1,  # 32 tokens for item 2
    1,1,1,...,1,  # 32 tokens for item 3
    0,0,0,...,0,  # 32 tokens for padding
    0,0,0,...,0   # 32 tokens for padding
]
```

### Item Representation Extraction

We use the **last token** of each item's token sequence as the item representation:

```python
# hidden_states: (batch, seq_len, n_codebook, n_embd)
item_repr = hidden_states[:, :, -1, :]  # Take position 31 of each item

# Alternative strategies (not implemented):
# - First token: hidden_states[:, :, 0, :]
# - Mean pooling: hidden_states.mean(dim=2)
# - Max pooling: hidden_states.max(dim=2)[0]
```

## Future Improvements

1. **Pooling strategies**: Try different ways to extract item representation
2. **Position embeddings**: Add explicit token position embeddings within each item
3. **Hierarchical attention**: Separate intra-item and inter-item attention
4. **Sparse attention**: Use patterns like Longformer to reduce computation

## Comparison with Original RPG

| Aspect | Original RPG | Token-Level |
|--------|--------------|-------------|
| **Sequence length** | seq_len | seq_len × 32 |
| **Attention computation** | O(seq_len²) | O((seq_len×32)²) |
| **Memory** | Low | High |
| **Expressiveness** | Medium | High |
| **Training speed** | Fast | Slow |
| **Inference speed** | Fast | Medium |
| **Generation** | Parallel ✅ | Parallel ✅ |

