# MHL (Masked Hierarchical Learning) Model

## Overview

MHL is a novel recommendation model that combines the efficiency of RPG with the benefits of masked language modeling. It extends RPG by introducing token-level masking during training and reconstruction loss, enabling the model to learn better representations through self-supervision.

## Key Features

### 1. **Masked Training Strategy**
- **Random Masking**: During training, randomly mask 15% of tokens in the input sequence
- **Reconstruction Task**: Predict masked tokens to reconstruct the original sequence
- **Self-Supervised Learning**: Learn better item representations through reconstruction

### 2. **Dual Loss Function**
- **Reconstruction Loss**: Predict masked tokens (similar to BERT's MLM)
- **Next-Item Prediction Loss**: Predict next items (original RPG objective)
- **Combined Training**: Both losses are optimized simultaneously

### 3. **Item-Level Encoding**
- **Efficiency**: Uses item-level encoding like RPG (average pooling of tokens)
- **Scalability**: Maintains computational efficiency while adding masking benefits
- **Compatibility**: Compatible with existing RPG infrastructure

## Architecture

```
Input Items → Token IDs → Masking → Item-Level Embeddings → GPT2 → Prediction Heads
     ↓              ↓         ↓              ↓              ↓         ↓
Original    Semantic    Randomly    Average Pool    Causal     32 Heads
Items       Tokens      Masked      to Items        Attention  (per codebook)
```

## Key Differences from RPG

| Aspect | RPG | MHL |
|--------|-----|-----|
| Training | Next-item prediction only | Next-item + Reconstruction |
| Masking | No masking | Random token masking |
| Loss | Single loss | Dual loss |
| Self-supervision | No | Yes (reconstruction) |
| Efficiency | High | High (same as RPG) |

## Configuration

Key parameters in `config.yaml`:

```yaml
# Masking configuration
mask_ratio: 0.15  # Mask 15% of tokens during training

# Loss weights
recon_loss_weight: 1.0      # Weight for reconstruction loss
next_item_loss_weight: 1.0  # Weight for next-item prediction loss

# Architecture (same as RPG)
n_embd: 768
n_layer: 6
n_head: 12
```

## Usage

### Training
```python
from genrec.models import MHL

model = MHL(config, dataset, tokenizer)
outputs = model(batch, return_loss=True)
loss = outputs.loss  # Combined reconstruction + next-item loss
```

### Inference
```python
model.eval()
with torch.no_grad():
    generated_tokens, _ = model.generate(batch, n_return_sequences=1)
```

## Benefits

1. **Better Representations**: Masking forces the model to learn robust item representations
2. **Self-Supervision**: Reconstruction task provides additional supervision signal
3. **Efficiency**: Maintains RPG's computational efficiency
4. **Flexibility**: Can adjust masking ratio and loss weights
5. **Compatibility**: Easy to integrate with existing RPG codebase

## Expected Improvements

- **Better Generalization**: Masking helps model generalize to unseen item combinations
- **Robust Representations**: Learned representations should be more robust to noise
- **Improved Performance**: Dual loss should improve recommendation quality
- **Faster Convergence**: Additional supervision signal may speed up training

## Testing

Run the test script to verify model functionality:

```bash
python test_mhl.py
```

This will test:
- Model initialization
- Forward pass with masking
- Loss computation
- Generation capability

## Future Work

1. **Adaptive Masking**: Dynamic masking ratio based on training progress
2. **Advanced Masking Strategies**: Token-level vs item-level masking comparison
3. **Loss Weight Scheduling**: Dynamic loss weight adjustment during training
4. **Multi-Task Learning**: Additional auxiliary tasks for better representations
