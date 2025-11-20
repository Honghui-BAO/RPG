# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for the vocab tokenizer.

Usage:
    python train_vocab_tokenizer.py --model RPG --dataset AmazonReviews2014 --vocab_epochs 100
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RPG', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--vocab_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--vocab_batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--vocab_lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--vocab_hidden_dim', type=int, default=512, help='Hidden dimension for MLPs')
    parser.add_argument('--vocab_num_layers', type=int, default=2, help='Number of layers in MLPs')
    parser.add_argument('--vocab_dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    return parser.parse_known_args()


def main():
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    
    # Override config with vocab settings
    config_dict = {
        **command_line_configs,
        'tokenizer_type': 'vocab',
        'vocab_epochs': args.vocab_epochs,
        'vocab_batch_size': args.vocab_batch_size,
        'vocab_lr': args.vocab_lr,
        'vocab_hidden_dim': args.vocab_hidden_dim,
        'vocab_num_layers': args.vocab_num_layers,
        'vocab_dropout': args.vocab_dropout,
        'device': args.device,
    }
    
    print(f"[TRAIN] Training vocab tokenizer with config:")
    print(f"  - Epochs: {args.vocab_epochs}")
    print(f"  - Batch size: {args.vocab_batch_size}")
    print(f"  - Learning rate: {args.vocab_lr}")
    print(f"  - Hidden dim: {args.vocab_hidden_dim}")
    print(f"  - Num layers: {args.vocab_num_layers}")
    print(f"  - Dropout: {args.vocab_dropout}")
    print(f"  - Device: {args.device}")
    
    # Initialize pipeline (this will load the dataset)
    print(f"\n[TRAIN] Initializing pipeline...")
    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=None,
        config_dict=config_dict
    )
    
    # Get the tokenizer (with initialized VocabTokenizer but untrained)
    tokenizer = pipeline.tokenizer
    
    if tokenizer.tokenizer_type != 'vocab':
        raise ValueError(f"Expected vocab tokenizer, got {tokenizer.tokenizer_type}")
    
    if not hasattr(tokenizer, 'vocab_tokenizer'):
        raise ValueError("VocabTokenizer not initialized!")
    
    vocab_tokenizer = tokenizer.vocab_tokenizer
    dataset = pipeline.raw_dataset  # Pipeline has raw_dataset, not dataset
    
    # Load T5 embeddings for training
    print(f"\n[TRAIN] Loading T5 embeddings...")
    sent_emb_path = os.path.join(
        dataset.cache_dir, 'processed',
        f'{os.path.basename(tokenizer.config["sent_emb_model"])}.sent_emb'
    )
    
    if not os.path.exists(sent_emb_path):
        raise FileNotFoundError(f"Sentence embeddings not found at {sent_emb_path}. Please run tokenizer initialization first.")
    
    sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, tokenizer.config['sent_emb_dim'])
    print(f"[TRAIN] Loaded embeddings shape: {sent_embs.shape}")
    
    # Convert to torch tensor
    sent_embs_tensor = torch.from_numpy(sent_embs).to(args.device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(vocab_tokenizer.parameters(), lr=args.vocab_lr)
    
    # Train the vocab tokenizer
    print(f"\n[TRAIN] Starting training...")
    print(f"=" * 80)
    
    losses = vocab_tokenizer.train_tokenizer(
        embeddings=sent_embs_tensor,
        optimizer=optimizer,
        n_epochs=args.vocab_epochs,
        batch_size=args.vocab_batch_size
    )
    
    print(f"=" * 80)
    print(f"[TRAIN] Training completed!")
    print(f"[TRAIN] Final loss: {losses[-1]:.6f}")
    
    # Save the trained model
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            dataset.cache_dir, 'processed'
        )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'{os.path.basename(tokenizer.config["sent_emb_model"])}_vocab_tokenizer.pt'
    )
    
    print(f"\n[TRAIN] Saving model to {checkpoint_path}...")
    torch.save({
        'model_state_dict': vocab_tokenizer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': {
            'vocab_hidden_dim': args.vocab_hidden_dim,
            'vocab_num_layers': args.vocab_num_layers,
            'vocab_dropout': args.vocab_dropout,
            'vocab_epochs': args.vocab_epochs,
            'vocab_batch_size': args.vocab_batch_size,
            'vocab_lr': args.vocab_lr,
        }
    }, checkpoint_path)
    
    print(f"[TRAIN] Model saved successfully!")
    
    # Generate 4-token representations
    print(f"\n[TRAIN] Generating 4-token representations for all items...")
    vocab_tokenizer.eval()
    with torch.no_grad():
        token_indices = vocab_tokenizer.tokenize_items(sent_embs_tensor)
    
    # Save token representations
    token_path = os.path.join(
        checkpoint_dir,
        f'{os.path.basename(tokenizer.config["sent_emb_model"])}_vocab_tokens.json'
    )
    
    import json
    item2tokens = {}
    for i in range(token_indices.shape[0]):
        item = tokenizer.id2item[i + 1]
        tokens = tuple((token_indices[i] + 1).cpu().numpy().tolist())
        item2tokens[item] = tokens
    
    print(f"[TRAIN] Saving token representations to {token_path}...")
    with open(token_path, 'w') as f:
        json.dump(item2tokens, f)
    
    print(f"[TRAIN] Token representations saved!")
    print(f"\n[TRAIN] Example tokenizations (first 5 items):")
    for idx, (item, tokens) in enumerate(list(item2tokens.items())[:5]):
        print(f"  {item}: {tokens}")
    
    print(f"\n[TRAIN] Done! You can now train RPG with vocab tokenizer.")
    print(f"[TRAIN] Set tokenizer_type='vocab' and vocab_checkpoint='{checkpoint_path}' in config.")


if __name__ == '__main__':
    main()
