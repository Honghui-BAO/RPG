#!/usr/bin/env python3
"""
Test script for embedding similarity inference mode
"""

import argparse
import torch
from torch.utils.data import DataLoader

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RPG', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--category', type=str, default='Beauty', help='Dataset category')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Which split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--topk', type=int, default=10, help='Top-k recommendations')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    
    # Add category to config
    command_line_configs['category'] = args.category
    command_line_configs['eval_batch_size'] = args.batch_size

    # Create pipeline with checkpoint
    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs
    )
    
    # Prepare dataloader for inference
    split = args.split
    dataloader = DataLoader(
        pipeline.tokenized_datasets[split],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[split]
    )
    
    # Prepare model and dataloader with accelerator
    pipeline.model, dataloader = pipeline.accelerator.prepare(
        pipeline.model, dataloader
    )
    
    # Set inference mode to direct (embedding similarity)
    pipeline.trainer.model.generate_w_decoding_graph = False
    pipeline.trainer.model.use_token_overlap = False
    
    # Run evaluation
    pipeline.log(f'Running embedding similarity inference on {split} set...')
    pipeline.log(f'Category: {args.category}')
    pipeline.log(f'Batch size: {args.batch_size}')
    pipeline.log(f'Top-k: {args.topk}')
    
    results = pipeline.trainer.evaluate(dataloader, split=split)
    
    if pipeline.accelerator.is_main_process:
        pipeline.log(f'\n{split.upper()} Results (Embedding Similarity):')
        for key, value in results.items():
            pipeline.log(f'  {key}: {value:.4f}')
    
    pipeline.trainer.end()
