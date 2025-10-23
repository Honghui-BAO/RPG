#!/usr/bin/env python3
"""
Test script for MHL model
"""

import torch
import yaml
from genrec.models import MHL
from genrec.data import RecDataset, OPQTokenizer

def test_mhl_model():
    """Test MHL model initialization and forward pass"""
    
    # Load config
    with open('genrec/models/MHL/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy dataset and tokenizer
    print("Creating dummy dataset and tokenizer...")
    
    # Dummy dataset (you might need to adjust this based on your actual dataset)
    class DummyDataset:
        def __init__(self):
            self.n_items = 1000
            self.item2id = {f"item_{i}": i for i in range(self.n_items)}
            self.id2item = {i: f"item_{i}" for i in range(self.n_items)}
    
    # Dummy tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.vocab_size = 8195  # 1 (PAD) + 8192 (tokens) + 1 (EOS) + 1 (MASK)
            self.n_digit = 32
            self.eos_token = 8193
            self.mask_token_id = 8194
            self.item2tokens = {f"item_{i}": [j + 1 + (k * 256) for k in range(32) for j in range(8)] for i in range(1000)}
    
    dataset = DummyDataset()
    tokenizer = DummyTokenizer()
    
    # Create model
    print("Creating MHL model...")
    model = MHL(config, dataset, tokenizer)
    print(f"Model created successfully!")
    print(f"Model parameters: {model.n_parameters}")
    
    # Create dummy batch
    batch_size = 4
    seq_len = 10
    
    batch = {
        'input_ids': torch.randint(1, dataset.n_items, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'labels': torch.randint(1, dataset.n_items, (batch_size, 1)),
        'seq_lens': torch.full((batch_size,), seq_len)
    }
    
    print(f"Batch shape: input_ids={batch['input_ids'].shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    model.train()  # Set to training mode to enable masking
    outputs = model(batch, return_loss=True)
    
    print(f"Forward pass successful!")
    print(f"Output loss: {outputs.loss.item():.4f}")
    print(f"Final states shape: {outputs.final_states.shape}")
    
    # Test generation
    print("Testing generation...")
    model.eval()  # Set to eval mode to disable masking
    with torch.no_grad():
        generated_tokens, visited_counts = model.generate(batch, n_return_sequences=1)
        print(f"Generation successful!")
        print(f"Generated tokens shape: {generated_tokens.shape}")
        print(f"Sample generated tokens: {generated_tokens[0, 0, :5]}")
    
    print("All tests passed! ✅")

if __name__ == "__main__":
    test_mhl_model()
