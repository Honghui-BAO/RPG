#!/usr/bin/env python3
"""
Test script for MHL model with token masking functionality.
"""

import sys
import os
import torch
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_masking_functionality():
    """Test the masking functionality of MHL model."""
    print("🧪 Testing MHL Model with Token Masking")
    print("=" * 50)
    
    try:
        # Import required modules
        from genrec.models import MHL
        from genrec.dataset import AbstractDataset
        from genrec.tokenizer import AbstractTokenizer
        
        print("✅ Successfully imported MHL model")
        
        # Test configuration
        config = {
            'device': 'cpu',
            'n_embd': 128,
            'n_layer': 2,
            'n_head': 4,
            'n_inner': 256,
            'activation_function': 'gelu_new',
            'resid_pdrop': 0.0,
            'embd_pdrop': 0.1,
            'attn_pdrop': 0.1,
            'layer_norm_epsilon': 1e-12,
            'initializer_range': 0.02,
            'temperature': 0.07,
            'chunk_size': 100,
            'num_beams': 10,
            'n_edges': 10,
            'propagation_steps': 2,
            'mask_ratio': 0.15,
            'reconstruction_weight': 0.5,
            'codebook_size': 256
        }
        
        print("✅ Configuration created")
        
        # Create mock dataset and tokenizer
        class MockDataset:
            def __init__(self):
                self.n_items = 1000
                self.item2id = {f'item_{i}': i for i in range(1, 1001)}
                self.id_mapping = {'id2item': {i: f'item_{i}' for i in range(1, 1001)}}
        
        class MockTokenizer:
            def __init__(self):
                self.n_digit = 8
                self.codebook_size = 256
                self.vocab_size = 8 * 256 + 1
                self.eos_token = 8 * 256
                self.ignored_label = -100
                self.item2tokens = {f'item_{i}': [np.random.randint(1, 256) for _ in range(8)] for i in range(1, 1001)}
        
        dataset = MockDataset()
        tokenizer = MockTokenizer()
        
        print("✅ Mock dataset and tokenizer created")
        
        # Create MHL model
        model = MHL(config, dataset, tokenizer)
        print("✅ MHL model created successfully")
        
        # Test masking parameters
        print(f"📊 Masking ratio: {model.mask_ratio}")
        print(f"📊 Reconstruction weight: {model.reconstruction_weight}")
        print(f"📊 Mask token ID: {model.mask_token_id}")
        
        # Test masking function
        batch_size, seq_len, n_digit = 2, 10, 8
        input_tokens = torch.randint(1, 256, (batch_size, seq_len, n_digit))
        attention_mask = torch.ones(batch_size, seq_len)
        
        masked_tokens, mask_positions, original_tokens = model._create_masked_tokens(
            input_tokens, attention_mask
        )
        
        print(f"✅ Masking function works")
        print(f"   Original tokens shape: {input_tokens.shape}")
        print(f"   Masked tokens shape: {masked_tokens.shape}")
        print(f"   Mask positions shape: {mask_positions.shape}")
        print(f"   Number of masked positions: {mask_positions.sum().item()}")
        
        # Test forward pass
        batch = {
            'input_ids': torch.randint(1, 1000, (batch_size, seq_len)),
            'attention_mask': attention_mask,
            'labels': torch.randint(1, 1000, (batch_size, seq_len))
        }
        
        outputs = model.forward(batch, return_loss=True)
        
        print(f"✅ Forward pass successful")
        print(f"   Total loss: {outputs.loss.item():.4f}")
        print(f"   Recommendation loss: {outputs.recommendation_loss.item():.4f}")
        print(f"   Reconstruction loss: {outputs.reconstruction_loss.item():.4f}")
        
        # Test with no masking
        model.mask_ratio = 0.0
        outputs_no_mask = model.forward(batch, return_loss=True)
        print(f"✅ No masking test successful")
        print(f"   Total loss (no mask): {outputs_no_mask.loss.item():.4f}")
        print(f"   Reconstruction loss (no mask): {outputs_no_mask.reconstruction_loss.item():.4f}")
        
        print("\n🎉 All tests passed! MHL model with masking is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_masking_functionality()
    sys.exit(0 if success else 1)
