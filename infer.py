# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    parser.add_argument('--infer_mode', type=str, default='direct', 
                        choices=['direct', 'graph', 'overlap'],
                        help='Inference mode: direct (embedding matching), graph (graph-constrained), overlap (token overlap counting)')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Which split to evaluate')
    
    # Deprecated flag for backward compatibility
    parser.add_argument('--use_graph', action='store_true', help='[Deprecated] Use --infer_mode=graph instead')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

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
        batch_size=pipeline.config['eval_batch_size'],
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[split]
    )
    
    # Prepare model and dataloader with accelerator
    pipeline.model, dataloader = pipeline.accelerator.prepare(
        pipeline.model, dataloader
    )
    
    # Set inference mode (handle backward compatibility)
    if args.use_graph:
        pipeline.log('[Warning] --use_graph is deprecated. Use --infer_mode=graph instead.')
        infer_mode = 'graph'
    else:
        infer_mode = args.infer_mode
    
    # Configure model based on inference mode
    if infer_mode == 'graph':
        pipeline.trainer.model.generate_w_decoding_graph = True
        pipeline.trainer.model.use_token_overlap = False
        pipeline.log('Using graph-constrained decoding')
    elif infer_mode == 'overlap':
        pipeline.trainer.model.generate_w_decoding_graph = False
        pipeline.trainer.model.use_token_overlap = True
        pipeline.log('Using token overlap counting')
    else:  # 'direct'
        pipeline.trainer.model.generate_w_decoding_graph = False
        pipeline.trainer.model.use_token_overlap = False
        pipeline.log('Using direct embedding matching')
        pipeline.log(f'  - normalize_embeddings: {pipeline.config.get("normalize_embeddings", True)}')
        pipeline.log(f'  - similarity_temperature: {pipeline.config.get("similarity_temperature", 1.0)}')
    
    # Run evaluation
    pipeline.log(f'Running inference on {split} set...')
    pipeline.log(f'Inference mode: {infer_mode}')
    
    results = pipeline.trainer.evaluate(dataloader, split=split)
    
    if pipeline.accelerator.is_main_process:
        pipeline.log(f'\n{split.upper()} Results:')
        for key, value in results.items():
            pipeline.log(f'  {key}: {value:.4f}')
    
    pipeline.trainer.end()

