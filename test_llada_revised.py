#!/usr/bin/env python3
"""
Test script for LLADA Revised model
"""

import torch
from genrec.pipeline import Pipeline

def test_llada_revised():
    """Test LLADA Revised model initialization and forward pass"""
    
    # Create a simple config
    config_dict = {
        'category': 'Sports_and_Outdoors',
        'model': 'LLADA_REVISED',
        'dataset': 'AmazonReviews2014',
        'n_embd': 128,  # Smaller for testing
        'n_layer': 1,
        'n_head': 2,
        'n_inner': 256,
        'max_item_seq_len': 10,
        'diffusion_steps': 8,
        'temperature': 0.1,
        'train_batch_size': 2,
        'eval_batch_size': 2,
        'device': 'cpu',  # Use CPU for testing
        'use_ddp': False,
        'rand_seed': 42,
        'reproducibility': True,
        'tensorboard_log_dir': './logs',
        'ckpt_dir': './checkpoints',
        'metrics': ['recall', 'ndcg'],
        'topk': [10, 20],
        'val_metric': 'recall@10',
        'eval_interval': 1,
        'patience': None,
        'max_grad_norm': 1.0,
        'lr': 0.001,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'num_proc': 1,
    }
    
    try:
        print("Creating pipeline...")
        pipeline = Pipeline(
            model_name='LLADA_REVISED',
            dataset_name='AmazonReviews2014',
            config_dict=config_dict
        )
        
        print("Pipeline created successfully!")
        print(f"Model: {pipeline.model}")
        print(f"Tokenizer: {pipeline.tokenizer}")
        print(f"Model parameters: {pipeline.model.n_parameters}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        
        # Create a dummy batch
        batch = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 2 samples, 3 items each
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]]),
            'labels': torch.tensor([[7], [8]]),  # Target items
            'seq_lens': torch.tensor([3, 3])
        }
        
        # Move to device
        device = pipeline.config['device']
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        pipeline.model.eval()
        with torch.no_grad():
            outputs = pipeline.model(batch, return_loss=True)
            print(f"Forward pass successful! Loss: {outputs.loss.item():.4f}")
        
        # Test generation
        print("\nTesting generation...")
        with torch.no_grad():
            preds = pipeline.model.generate(batch, n_return_sequences=5)
            print(f"Generation successful! Predictions shape: {preds.shape}")
            print(f"Sample predictions: {preds[0, :5, 0].tolist()}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llada_revised()
