#!/usr/bin/env python3
"""
Test script for MHL model with token-level masking functionality.
"""

import sys
import os
import torch
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_token_level_masking():
    """Test MHL model with token-level masking."""
    print("🧪 Testing MHL Model with Token-Level Masking")
    print("=" * 60)
    
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
            'mask_ratio': 0.15,  # Token-level masking ratio
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
        
        # Test token-level masking
        batch_size, seq_len, n_digit = 2, 5, 8
        input_tokens = torch.randint(1, 256, (batch_size, seq_len, n_digit))
        attention_mask = torch.ones(batch_size, seq_len)
        
        print(f"\n🔍 Testing token-level masking:")
        print(f"   Input shape: {input_tokens.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        print(f"   Mask ratio: {model.mask_ratio}")
        
        # Test masking function
        masked_tokens, mask_positions, original_tokens = model._create_masked_tokens(
            input_tokens, attention_mask
        )
        
        print(f"✅ Token-level masking function works")
        print(f"   Masked tokens shape: {masked_tokens.shape}")
        print(f"   Mask positions shape: {mask_positions.shape}")
        print(f"   Original tokens shape: {original_tokens.shape}")
        print(f"   Total masked tokens: {mask_positions.sum().item()}")
        print(f"   Mask ratio achieved: {mask_positions.sum().item() / mask_positions.numel():.3f}")
        
        # Test forward pass
        batch = {
            'input_ids': torch.randint(1, 1000, (batch_size, seq_len)),
            'attention_mask': attention_mask,
            'labels': torch.randint(1, 1000, (batch_size, seq_len))
        }
        
        print(f"\n🧪 Testing forward pass with token-level masking...")
        outputs = model.forward(batch, return_loss=True)
        
        print(f"✅ Forward pass completed successfully")
        print(f"   Total loss: {outputs.loss.item():.4f}")
        print(f"   Recommendation loss: {outputs.recommendation_loss.item():.4f}")
        print(f"   Reconstruction loss: {outputs.reconstruction_loss.item():.4f}")
        
        # Test with no masking
        model.mask_ratio = 0.0
        outputs_no_mask = model.forward(batch, return_loss=True)
        print(f"✅ No masking test successful")
        print(f"   Total loss (no mask): {outputs_no_mask.loss.item():.4f}")
        print(f"   Reconstruction loss (no mask): {outputs_no_mask.reconstruction_loss.item():.4f}")
        
        # Test different mask ratios
        print(f"\n🔬 Testing different mask ratios:")
        for mask_ratio in [0.05, 0.1, 0.2, 0.3]:
            model.mask_ratio = mask_ratio
            masked_tokens, mask_positions, _ = model._create_masked_tokens(input_tokens, attention_mask)
            actual_ratio = mask_positions.sum().item() / mask_positions.numel()
            print(f"   Target: {mask_ratio:.2f}, Actual: {actual_ratio:.3f}, Masked tokens: {mask_positions.sum().item()}")
        
        print("\n🎉 All token-level masking tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_token_level_masking()
    sys.exit(0 if success else 1)
