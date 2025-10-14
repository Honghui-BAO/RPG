# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LLaDA-style Diffusion Model for Sequential Recommendation

Reference: Large Language Diffusion Models (https://arxiv.org/abs/2502.09992)

Key differences from autoregressive RPG:
- Uses mask-based diffusion instead of next-token prediction
- Predicts all codes in parallel rather than sequentially
- Training: sample timestep t, mask codes, predict masked positions
- Inference: iterative denoising from fully masked to clean codes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import numpy as np

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """Residual Block for prediction heads"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class LLaDARecommender(AbstractModel):
    """
    LLaDA-style diffusion model for recommendation
    
    Forward process: Progressively mask target item's semantic codes
    Reverse process: Transformer predicts masked codes conditioned on timestep
    """
    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(LLaDARecommender, self).__init__(config, dataset, tokenizer)
        
        # Semantic ID mapping
        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        
        # Diffusion parameters
        self.T = config.get('diffusion_steps', 32)  # Total diffusion steps
        self.mask_schedule = config.get('mask_schedule', 'linear')  # linear, cosine, square
        self.codes_per_step = config.get('codes_per_step', None)  # If set, determine this many codes per step
        
        # Special tokens
        self.mask_token_id = tokenizer.mask_token_id
        
        # BERT backbone (bidirectional attention for better sequence understanding)
        bert_config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=tokenizer.max_token_seq_len + 1,
            hidden_size=config['n_embd'],
            num_hidden_layers=config['n_layer'],
            num_attention_heads=config['n_head'],
            intermediate_size=config['n_inner'],
            hidden_act=config['activation_function'],
            hidden_dropout_prob=config['embd_pdrop'],
            attention_probs_dropout_prob=config['attn_pdrop'],
            layer_norm_eps=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
        )
        self.encoder = BertModel(bert_config)
        
        # Time step embedding
        self.time_embed = nn.Embedding(self.T + 1, config['n_embd'])
        
        # Prediction heads (32 heads for 32 semantic codes)
        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)
        
        # Loss function
        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        
        # For graph-constrained decoding (inherited from RPG)
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config.get('chunk_size', 1024)
        self.num_beams = config.get('num_beams', 50)
        self.n_edges = config.get('n_edges', 50)
        self.propagation_steps = config.get('propagation_steps', 3)

    def _map_item_tokens(self) -> torch.Tensor:
        """Maps item IDs to their semantic code tokens"""
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.encoder.get_input_embeddings().parameters())
        time_emb_params = sum(p.numel() for p in self.time_embed.parameters())
        return f'#Embedding parameters: {emb_params}\n' \
               f'#Time embedding parameters: {time_emb_params}\n' \
               f'#Non-embedding parameters: {total_params - emb_params - time_emb_params}\n' \
               f'#Total trainable parameters: {total_params}\n'

    def get_mask_ratio(self, t: int) -> float:
        """
        Get masking ratio for timestep t
        
        Args:
            t: timestep (1 to T)
        
        Returns:
            mask_ratio: proportion of codes to mask (0 to 1)
        """
        if self.mask_schedule == 'linear':
            return t / self.T
        elif self.mask_schedule == 'cosine':
            return np.cos(np.pi * t / (2 * self.T))
        elif self.mask_schedule == 'square':
            return (t / self.T) ** 2
        else:
            return t / self.T

    def forward_diffusion(self, target_codes: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion process: mask target codes
        
        Args:
            target_codes: (batch_size, n_digit) clean semantic codes
            t: (batch_size,) timesteps
        
        Returns:
            masked_codes: (batch_size, n_digit) codes with some positions masked
            mask: (batch_size, n_digit) boolean mask indicating which positions are masked
        """
        batch_size, n_digit = target_codes.shape
        device = target_codes.device
        
        masked_codes = target_codes.clone()
        mask = torch.zeros_like(target_codes, dtype=torch.bool)
        
        for b in range(batch_size):
            mask_ratio = self.get_mask_ratio(t[b].item())
            num_masked = int(n_digit * mask_ratio)
            
            if num_masked > 0:
                # Randomly select positions to mask
                masked_positions = torch.randperm(n_digit, device=device)[:num_masked]
                masked_codes[b, masked_positions] = self.mask_token_id
                mask[b, masked_positions] = True
        
        return masked_codes, mask

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        """
        Forward pass with diffusion training
        
        Args:
            batch: dict with keys ['input_ids', 'attention_mask', 'labels', 'seq_lens']
        
        Returns:
            outputs with loss if return_loss=True
        """
        batch_size = batch['input_ids'].shape[0]
        device = batch['input_ids'].device
        
        # Get valid labels (filter out -100 and 0 which is padding)
        labels_flat = batch['labels'].view(-1)
        label_mask = (labels_flat != -100) & (labels_flat > 0)
        valid_labels = labels_flat[label_mask]
        
        # Get number of valid labels per batch
        num_valid = label_mask.view(batch_size, -1).sum(dim=1)
        
        # Sample timesteps for each valid label
        t = torch.randint(1, self.T + 1, (valid_labels.shape[0],), device=device)
        
        # Get target item codes for valid labels
        target_codes = self.item_id2tokens[valid_labels]  # (num_valid_labels, n_digit)
        
        # TODO: Enable forward diffusion after basic training works
        # Forward diffusion: mask some codes
        # masked_codes, mask = self.forward_diffusion(target_codes, t)
        # For now, skip masking to debug
        
        # For LLADA, we use the same architecture as RPG but predict masked tokens
        # Get embeddings for history items
        input_tokens = self.item_id2tokens[batch['input_ids']]  # (batch_size, seq_len, n_digit)
        input_embs = self.encoder.embeddings.word_embeddings(input_tokens).mean(dim=-2)  # (batch_size, seq_len, n_embd)
        
        # Pass through BERT (bidirectional attention - better for recommendation)
        outputs = self.encoder(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        
        # Get representations for all positions (like RPG)
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)  # (batch_size, seq_len, n_digit, n_embd)
        
        outputs.final_states = final_states
        
        if return_loss:
            # Extract states for positions with valid labels
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            # selected_states shape: (num_valid_labels, n_digit, n_embd)
            
            # Add time embedding to selected states
            time_emb = self.time_embed(t).unsqueeze(1)  # (num_valid_labels, 1, n_embd)
            selected_states = selected_states + time_emb  # Broadcast across n_digit dimension
            
            # Normalize states
            selected_states_norm = F.normalize(selected_states, dim=-1)
            
            # Get token embeddings (exclude PAD, EOS, and MASK)
            # For LLADA: vocab is [PAD, codes_1-8192, EOS, MASK]
            # We only want codes_1-8192 (the 32*256=8192 semantic tokens)
            token_emb = self.encoder.embeddings.word_embeddings.weight[1:1+self.n_pred_head*self.config['codebook_size']]
            token_emb_norm = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb_norm, self.n_pred_head, dim=0)
            
            # Compute loss (same as RPG, simplified for debugging)
            selected_states_chunks = torch.chunk(selected_states_norm, self.n_pred_head, dim=1)
            token_labels = self.item_id2tokens[valid_labels]  # (num_valid_labels, n_digit)
            
            losses = []
            for i in range(self.n_pred_head):
                # Compute logits for digit i
                logits = torch.matmul(selected_states_chunks[i].squeeze(dim=1), token_embs[i].T) / self.temperature
                
                # Get labels - exactly like RPG
                labels = token_labels[:, i] - i * self.config['codebook_size'] - 1
                
                # Compute loss
                loss_i = self.loss_fct(logits, labels)
                losses.append(loss_i)
            
            outputs.loss = torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=device)
        
        return outputs

    def check_code_validity(self, codes: torch.Tensor) -> dict:
        """
        Check if generated codes match any real items
        
        Args:
            codes: (batch_size, n_digit) generated semantic codes
            
        Returns:
            dict with statistics about code validity
        """
        batch_size = codes.shape[0]
        all_item_codes = self.item_id2tokens[1:]  # (n_items-1, n_digit)
        
        exact_matches = 0
        max_matches = []
        
        for b in range(batch_size):
            # Check if this code combination exists in item library
            matches = (codes[b].unsqueeze(0) == all_item_codes).all(dim=1)
            
            if matches.any():
                exact_matches += 1
            
            # Count maximum matching codes
            num_matching_codes = (codes[b].unsqueeze(0) == all_item_codes).sum(dim=1)
            max_matches.append(num_matching_codes.max().item())
        
        return {
            'exact_match_rate': exact_matches / batch_size,
            'avg_max_matching_codes': sum(max_matches) / len(max_matches),
            'exact_matches': exact_matches,
            'total': batch_size
        }

    @torch.no_grad()
    def generate(self, batch, n_return_sequences=1, return_codes=False):
        """
        Generate recommendations using iterative denoising
        
        Args:
            batch: dict with user history
            n_return_sequences: number of items to recommend
            return_codes: if True, also return generated codes and validity stats
        
        Returns:
            predictions: (batch_size, n_return_sequences, 1)
            or (predictions, generated_codes, validity_stats) if return_codes=True
        """
        batch_size = batch['input_ids'].shape[0]
        device = batch['input_ids'].device
        
        # Initialize: all codes are masked
        current_codes = torch.full(
            (batch_size, self.n_pred_head),
            self.mask_token_id,
            device=device,
            dtype=torch.long
        )
        
        # Iterative denoising from t=T to t=1
        for t in reversed(range(1, self.T + 1)):
            # Early stopping: if all codes are determined, stop iterating
            if (current_codes != self.mask_token_id).all():
                break
            
            # Construct input
            input_tokens = self.item_id2tokens[batch['input_ids']]
            input_embs = self.encoder.embeddings.word_embeddings(input_tokens).mean(dim=-2)  # (batch, seq_len, n_embd)
            
            # Get embeddings for current codes (each item has 32 codes)
            # current_codes: (batch, 32)
            # after embedding: (batch, 32, n_embd)  
            # mean over 32 codes: (batch, n_embd)
            target_embs = self.encoder.embeddings.word_embeddings(current_codes).mean(dim=1, keepdim=True)  # (batch, 1, n_embd)
            all_embs = torch.cat([input_embs, target_embs], dim=1)  # (batch, seq_len+1, n_embd)
            
            # Add time embedding
            time_emb = self.time_embed(torch.full((batch_size,), t, device=device))
            all_embs[:, -1, :] = all_embs[:, -1, :] + time_emb
            
            # Extend attention mask
            attention_mask_extended = torch.cat([
                batch['attention_mask'],
                torch.ones(batch_size, 1, device=device, dtype=batch['attention_mask'].dtype)
            ], dim=1)
            
            # Forward through BERT
            outputs = self.encoder(inputs_embeds=all_embs, attention_mask=attention_mask_extended)
            target_hidden = outputs.last_hidden_state[:, -1, :]
            
            # Predict codes
            final_states = torch.cat([
                self.pred_heads[i](target_hidden).unsqueeze(1) 
                for i in range(self.n_pred_head)
            ], dim=1)
            
            # Get predictions for each digit
            final_states_norm = F.normalize(final_states, dim=-1)
            # Use correct token embedding range (exclude PAD, EOS, MASK)
            token_emb_norm = F.normalize(self.encoder.embeddings.word_embeddings.weight[1:1+self.n_pred_head*self.config['codebook_size']], dim=-1)
            token_embs = torch.chunk(token_emb_norm, self.n_pred_head, dim=0)
            
            predicted_codes = []
            confidence_scores = []
            
            for i in range(self.n_pred_head):
                logits = torch.matmul(final_states_norm[:, i, :], token_embs[i].T) / self.temperature
                probs = F.softmax(logits, dim=-1)
                
                # Get predicted code and confidence
                conf, pred_idx = probs.max(dim=-1)
                predicted_code = pred_idx + i * self.config['codebook_size'] + 1
                
                predicted_codes.append(predicted_code)
                confidence_scores.append(conf)
            
            predicted_codes = torch.stack(predicted_codes, dim=1)  # (batch, n_digit)
            confidence_scores = torch.stack(confidence_scores, dim=1)  # (batch, n_digit)
            
            # Update strategy: keep high-confidence predictions
            if t > 1:
                if self.codes_per_step is not None:
                    # Fixed number of codes per step (faster)
                    # Calculate how many codes to keep in total
                    steps_done = self.T - t + 1  # How many steps we've done
                    num_to_keep = min(steps_done * self.codes_per_step, self.n_pred_head)
                else:
                    # Original schedule-based approach
                    mask_ratio = self.get_mask_ratio(t - 1)
                    num_to_keep = int(self.n_pred_head * (1 - mask_ratio))
                
                # Keep top-k confident predictions
                _, top_k_indices = confidence_scores.topk(num_to_keep, dim=1)
                
                # Update current_codes (only change positions that are still MASK)
                for b in range(batch_size):
                    for idx in top_k_indices[b]:
                        # Only update if it's currently MASK (don't overwrite previous decisions)
                        if current_codes[b, idx] == self.mask_token_id:
                            current_codes[b, idx] = predicted_codes[b, idx]
            else:
                # Last step: use all predictions
                current_codes = predicted_codes
        
        # Convert codes to item IDs
        mapping_method = self.config.get('code_to_item_method', 'count')
        item_logits = self._codes_to_item_logits(current_codes, method=mapping_method)
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
        
        # Check code validity
        if return_codes:
            validity_stats = self.check_code_validity(current_codes)
            return preds.unsqueeze(-1), current_codes, validity_stats
        
        return preds.unsqueeze(-1)

    def _codes_to_item_logits(self, codes: torch.Tensor, method='count') -> torch.Tensor:
        """
        Convert semantic codes to item logits
        
        Args:
            codes: (batch_size, n_digit) semantic codes
            method: 'count', 'embedding', or 'hybrid'
        
        Returns:
            item_logits: (batch_size, n_items) logits for each item
        """
        batch_size = codes.shape[0]
        all_item_codes = self.item_id2tokens[1:]  # (n_items-1, n_digit)
        
        if method == 'count':
            # Method 1: Count matching codes (simple, current default)
            item_logits = torch.zeros(batch_size, self.dataset.n_items - 1, device=codes.device)
            for b in range(batch_size):
                matches = (codes[b].unsqueeze(0) == all_item_codes).sum(dim=1)
                item_logits[b] = matches.float()
        
        elif method == 'embedding':
            # Method 2: Use embedding similarity (more robust to mismatches)
            # Convert codes to embeddings
            code_embs = self.encoder.embeddings.word_embeddings(codes).mean(dim=1)  # (batch, n_embd)
            all_item_embs = self.encoder.embeddings.word_embeddings(all_item_codes).mean(dim=1)  # (n_items-1, n_embd)
            
            # Cosine similarity
            code_embs_norm = F.normalize(code_embs, dim=-1)
            all_item_embs_norm = F.normalize(all_item_embs, dim=-1)
            item_logits = torch.matmul(code_embs_norm, all_item_embs_norm.T)  # (batch, n_items-1)
        
        elif method == 'hybrid':
            # Method 3: Combine count and embedding (best of both)
            # First use count to filter top candidates
            matches_logits = torch.zeros(batch_size, self.dataset.n_items - 1, device=codes.device)
            for b in range(batch_size):
                matches = (codes[b].unsqueeze(0) == all_item_codes).sum(dim=1)
                matches_logits[b] = matches.float()
            
            # Then use embedding for final ranking among top candidates
            code_embs = self.encoder.embeddings.word_embeddings(codes).mean(dim=1)
            all_item_embs = self.encoder.embeddings.word_embeddings(all_item_codes).mean(dim=1)
            code_embs_norm = F.normalize(code_embs, dim=-1)
            all_item_embs_norm = F.normalize(all_item_embs, dim=-1)
            emb_logits = torch.matmul(code_embs_norm, all_item_embs_norm.T)
            
            # Weighted combination
            item_logits = 0.7 * matches_logits + 0.3 * emb_logits * 32  # Scale emb to same range
        
        return item_logits

    def calculate_loss(self, batch):
        """Required by AbstractModel"""
        return self.forward(batch, return_loss=True).loss

