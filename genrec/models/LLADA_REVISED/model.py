# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LLADA Revised: Non-causal GPT2 with item position embedding and target item masking

Key changes from original LLADA:
1. Remove causal attention (use bidirectional attention)
2. Add item position embedding (not token position)
3. Mask target item tokens during training
4. Predict only masked tokens in target item
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
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


class LLADARevised(AbstractModel):
    """
    LLADA Revised: RPG with iterative inference and MASK token
    
    Key features:
    - Causal attention (same as RPG)
    - Iterative inference with codes_per_step control
    - MASK token for inference (no diffusion training)
    - Simplified architecture - just RPG + iterative inference
    """
    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(LLADARevised, self).__init__(config, dataset, tokenizer)
        
        # Print all config information
        print("=" * 80)
        print("LLADA_REVISED Model Configuration:")
        print("=" * 80)
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=" * 80)
        
        # Semantic ID mapping
        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        
        # Diffusion parameters
        self.T = config.get('diffusion_steps', 32)
        self.mask_schedule = config.get('mask_schedule', 'linear')
        self.codes_per_step = config.get('codes_per_step', None)
        
        # Special tokens
        self.mask_token_id = tokenizer.mask_token_id
        
        # GPT2 backbone with causal attention (same as RPG)
        # Token-level encoding: n_positions = (max_item_seq_len + 1) * n_codebook
        # e.g., (50 + 1) * 32 = 1632 positions for token-level
        n_positions_token_level = (config['max_item_seq_len'] + 1) * config['n_codebook']
        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,  # This includes MASK token (8195)
            n_positions=n_positions_token_level,  # Expand for token-level encoding
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
            pad_token_id=0,  # Use 0 as padding token
            is_causal=True,  # Use causal attention like RPG
        )
        self.gpt2 = GPT2Model(gpt2config)
        
        # Item position embedding removed - keep it simple like RPG
        
        # Time step embedding removed - not needed without proper diffusion training
        
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
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters())
        return f'#Embedding parameters: {emb_params}\n' \
               f'#Non-embedding parameters: {total_params - emb_params}\n' \
               f'#Total trainable parameters: {total_params}\n'

    def get_mask_ratio(self, t: int) -> float:
        """Get masking ratio for timestep t"""
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
            p_mask: (batch_size, n_digit) probability of masking for loss weighting
        """
        batch_size, n_digit = target_codes.shape
        device = target_codes.device
        
        masked_codes = target_codes.clone()
        mask = torch.zeros_like(target_codes, dtype=torch.bool)
        p_mask = torch.zeros_like(target_codes, dtype=torch.float)
        
        for b in range(batch_size):
            mask_ratio = self.get_mask_ratio(t[b].item())
            num_masked = int(n_digit * mask_ratio)
            
            if num_masked > 0:
                # Randomly select positions to mask
                masked_positions = torch.randperm(n_digit, device=device)[:num_masked]
                masked_codes[b, masked_positions] = self.mask_token_id
                mask[b, masked_positions] = True
                p_mask[b, masked_positions] = mask_ratio
        
        return masked_codes, mask, p_mask

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        """
        Forward pass with target item masking
        
        Args:
            batch: dict with keys ['input_ids', 'attention_mask', 'labels', 'seq_lens']
        
        Returns:
            outputs with loss if return_loss=True
        """
        batch_size = batch['input_ids'].shape[0]
        device = batch['input_ids'].device
        
        # Get valid labels (filter out -100 and 0 which is padding)
        labels_flat = batch['labels'].view(-1)
        label_mask = (labels_flat != -100) & (labels_flat > 0) & (labels_flat < self.dataset.n_items)
        valid_labels = labels_flat[label_mask]
        
        # print(f"[DEBUG] batch_size: {batch_size}")
        # print(f"[DEBUG] labels_flat shape: {labels_flat.shape}, unique values: {torch.unique(labels_flat)}")
        # print(f"[DEBUG] valid_labels shape: {valid_labels.shape}, sample values: {valid_labels[:5] if len(valid_labels) > 0 else 'None'}")
        
        # Check if we have any valid labels
        if valid_labels.shape[0] == 0:
            # Return dummy loss if no valid labels
            outputs = type('Outputs', (), {})()
            outputs.loss = torch.tensor(0.0, device=device, requires_grad=True)
            return outputs
        
        # Sample timesteps for each valid label
        t = torch.randint(1, self.T + 1, (valid_labels.shape[0],), device=device)
        # print(f"[DEBUG] t shape: {t.shape}, sample values: {t[:5]}")
        
        # Get target item codes for valid labels
        target_codes = self.item_id2tokens[valid_labels]  # (num_valid_labels, n_digit)
        # print(f"[DEBUG] target_codes shape: {target_codes.shape}, sample values: {target_codes[:2, :5] if target_codes.shape[0] > 0 else 'None'}")
        
        # Forward diffusion: mask some codes
        masked_codes, mask, p_mask = self.forward_diffusion(target_codes, t)
        # print(f"[DEBUG] masked_codes shape: {masked_codes.shape}")
        # print(f"[DEBUG] mask shape: {mask.shape}, mask sum: {mask.sum()}")
        # print(f"[DEBUG] p_mask shape: {p_mask.shape}, p_mask range: [{p_mask.min():.3f}, {p_mask.max():.3f}]")
        
        # Token-level encoding: don't aggregate at item-level
        input_tokens = self.item_id2tokens[batch['input_ids']]  # (batch_size, seq_len, n_digit)
        
        # Ensure all tokens are within vocab range
        max_vocab_id = self.gpt2.config.vocab_size - 1
        input_tokens = torch.clamp(input_tokens, 0, max_vocab_id)
        
        # Debug: check token validity
        if (input_tokens < 0).any() or (input_tokens >= self.gpt2.config.vocab_size).any():
            print(f"[ERROR] Invalid input tokens detected!")
            print(f"  vocab_size: {self.gpt2.config.vocab_size}")
            print(f"  input_tokens min: {input_tokens.min()}, max: {input_tokens.max()}")
        
        # Get token embeddings without aggregation
        input_embs = self.gpt2.wte(input_tokens)  # (batch_size, seq_len, n_digit, n_embd)
        
        # Add item position embedding
        seq_lens = batch['seq_lens']
        item_positions = []
        for b in range(batch_size):
            pos = torch.arange(seq_lens[b], device=device)
            item_positions.append(pos)
        
        # Item position embedding removed - keep it simple like RPG
        
        # Add target item embeddings (masked)
        # Ensure masked_codes are within vocab range
        masked_codes = torch.clamp(masked_codes, 0, max_vocab_id)
        # Token-level: don't aggregate target codes either
        target_embs = self.gpt2.wte(masked_codes)  # (num_valid_labels, n_digit, n_embd)
        
        # Time embedding removed - not needed without proper diffusion training
        
        # Note: We don't add target position embedding to avoid sequence dependency
        # target_pos_emb = self.item_pos_embed(torch.full((valid_labels.shape[0],), max_len, device=device))
        # target_embs = target_embs + target_pos_emb.unsqueeze(1)
        
        # Reshape token-level embeddings and concatenate history and target
        # input_embs: (batch_size, seq_len, n_digit, n_embd)
        # target_embs: (num_valid_labels, n_digit, n_embd)
        
        seq_len_orig = input_embs.shape[1]
        n_digit = input_embs.shape[2]
        
        # Debug: check sequence length
        total_seq_len = seq_len_orig * n_digit + n_digit  # history + target
        print(f"[DEBUG] Token-level sequence length: {seq_len_orig} items * {n_digit} tokens + {n_digit} target = {total_seq_len} positions")
        print(f"[DEBUG] GPT2 n_positions: {self.gpt2.config.n_positions}")
        if total_seq_len > self.gpt2.config.n_positions:
            print(f"[WARNING] Sequence length {total_seq_len} exceeds GPT2 limit {self.gpt2.config.n_positions}!")
        
        # Reshape input to (batch_size, seq_len * n_digit, n_embd)
        input_embs_reshaped = input_embs.view(batch_size, seq_len_orig * n_digit, -1)
        
        all_embs = []
        for b in range(batch_size):
            # Find valid labels for this batch item
            batch_label_mask = label_mask.view(batch_size, -1)[b]
            if batch_label_mask.any():
                # Get target embeddings for this batch - only take the first valid target
                valid_idx = torch.where(batch_label_mask)[0]
                target_emb_b = target_embs[valid_idx[0]:valid_idx[0]+1]  # (1, n_digit, n_embd)
                all_emb_b = torch.cat([input_embs_reshaped[b:b+1], target_emb_b], dim=1)
            else:
                all_emb_b = input_embs_reshaped[b:b+1]
            all_embs.append(all_emb_b)
        
        # Pad to same length
        max_total_len = max(emb.shape[1] for emb in all_embs)
        padded_embs = []
        attention_masks = []
        
        for b in range(batch_size):
            emb = all_embs[b]
            if emb.shape[1] < max_total_len:
                pad_len = max_total_len - emb.shape[1]
                pad_emb = torch.zeros(1, pad_len, emb.shape[2], device=device)
                emb = torch.cat([emb, pad_emb], dim=1)
            
            # Create attention mask - expand for token-level
            attn_mask = torch.ones(1, emb.shape[1], device=device)
            # Find valid length for this batch item (in tokens)
            valid_len_items = seq_lens[b]
            valid_len_tokens = valid_len_items * n_digit
            # Add target tokens if present
            batch_label_mask = label_mask.view(batch_size, -1)[b]
            if batch_label_mask.any():
                valid_len_tokens += n_digit  # Add target item's tokens
            if valid_len_tokens < emb.shape[1]:
                attn_mask[0, valid_len_tokens:] = 0
            
            padded_embs.append(emb)
            attention_masks.append(attn_mask)
        
        all_embs = torch.cat(padded_embs, dim=0)  # (batch_size, max_total_len, n_embd)
        attention_mask = torch.cat(attention_masks, dim=0)  # (batch_size, max_total_len)
        
        # Ensure attention_mask has correct values (0 or 1)
        attention_mask = attention_mask.bool().float()
        
        # Pass through GPT2 (non-causal)
        outputs = self.gpt2(
            inputs_embeds=all_embs,
            attention_mask=attention_mask
        )
        
        # Get representations for target positions (token-level)
        target_hidden = []
        for b in range(batch_size):
            batch_label_mask = label_mask.view(batch_size, -1)[b]
            if batch_label_mask.any():
                # Target starts at position (seq_len * n_digit)
                target_start_pos = seq_lens[b] * n_digit
                # Extract n_digit tokens for the target
                target_tokens = outputs.last_hidden_state[b, target_start_pos:target_start_pos+n_digit, :]
                # Use the last token as the target representation
                target_hidden.append(target_tokens[-1:, :])
            else:
                # Dummy hidden state if no valid label
                target_hidden.append(torch.zeros(1, outputs.last_hidden_state.shape[-1], device=device))
        
        target_hidden = torch.cat(target_hidden, dim=0)  # (batch_size, n_embd)
        
        # Get representations for all codebook positions
        final_states = torch.cat([
            self.pred_heads[i](target_hidden).unsqueeze(1) 
            for i in range(self.n_pred_head)
        ], dim=1)  # (batch_size, n_digit, n_embd)
        
        outputs.final_states = final_states
        
        if return_loss:
            # Simple approach: compute loss directly on valid_labels
            # Get token embeddings (exclude PAD, EOS, and MASK)
            token_emb = self.gpt2.wte.weight[1:1+self.n_pred_head*self.config['codebook_size']]
            token_emb_norm = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb_norm, self.n_pred_head, dim=0)
            
            # Get target hidden states for each valid label
            target_hidden = []
            for b in range(batch_size):
                batch_label_mask = label_mask.view(batch_size, -1)[b]
                if batch_label_mask.any():
                    target_pos = seq_lens[b]  # Target is at position seq_len
                    target_hidden_b = outputs.last_hidden_state[b, target_pos:target_pos+1]  # (1, n_embd)
                    # Repeat for each valid label in this batch
                    valid_positions = torch.where(batch_label_mask)[0]
                    for _ in valid_positions:
                        target_hidden.append(target_hidden_b)
            
            if target_hidden:
                target_hidden = torch.cat(target_hidden, dim=0)  # (num_valid_labels, n_embd)
                # print(f"[DEBUG] target_hidden shape: {target_hidden.shape}")
                
                # Get representations for all codebook positions
                final_states = torch.cat([
                    self.pred_heads[i](target_hidden).unsqueeze(1) 
                    for i in range(self.n_pred_head)
                ], dim=1)  # (num_valid_labels, n_digit, n_embd)
                # print(f"[DEBUG] final_states shape: {final_states.shape}")
                
                # Normalize states
                final_states_norm = F.normalize(final_states, dim=-1)
                final_states_chunks = torch.chunk(final_states_norm, self.n_pred_head, dim=1)
                # print(f"[DEBUG] final_states_chunks[0] shape: {final_states_chunks[0].shape}")
                
                # Get token labels for valid labels
                token_labels = self.item_id2tokens[valid_labels]  # (num_valid_labels, n_digit)
                # print(f"[DEBUG] token_labels shape: {token_labels.shape}, sample values: {token_labels[:2, :5] if token_labels.shape[0] > 0 else 'None'}")
                
                losses = []
                for i in range(self.n_pred_head):
                    # Only compute loss for masked positions
                    mask_i = mask[:, i]  # Which samples have this codebook masked (num_valid_labels,)
                    # print(f"[DEBUG] Codebook {i}: mask_i shape: {mask_i.shape}, mask_i sum: {mask_i.sum()}")
                    
                    if mask_i.sum() > 0:
                        # Get masked logits and labels
                        masked_logits = torch.matmul(final_states_chunks[i].squeeze(dim=1), token_embs[i].T) / self.temperature
                        # print(f"[DEBUG] Codebook {i}: masked_logits before indexing shape: {masked_logits.shape}")
                        masked_logits = masked_logits[mask_i]  # (num_masked, codebook_size)
                        # print(f"[DEBUG] Codebook {i}: masked_logits after indexing shape: {masked_logits.shape}")
                        
                        labels = token_labels[:, i] - i * self.config['codebook_size'] - 1
                        masked_labels = labels[mask_i]  # (num_masked,)
                        masked_p_mask = p_mask[mask_i, i]  # (num_masked,)
                        # print(f"[DEBUG] Codebook {i}: masked_labels shape: {masked_labels.shape}, sample values: {masked_labels[:5]}")
                        # print(f"[DEBUG] Codebook {i}: masked_p_mask shape: {masked_p_mask.shape}, range: [{masked_p_mask.min():.3f}, {masked_p_mask.max():.3f}]")
                        
                        # Weight loss by p_mask (probability of masking)
                        token_loss = F.cross_entropy(masked_logits, masked_labels, reduction='none')
                        weighted_loss = token_loss / masked_p_mask
                        loss_i = torch.mean(weighted_loss)
                        # print(f"[DEBUG] Codebook {i}: loss_i: {loss_i.item():.4f}")
                        losses.append(loss_i)
                
                outputs.loss = torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=device)
                # print(f"[DEBUG] Final loss: {outputs.loss.item():.4f}")
            else:
                outputs.loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return outputs

    @torch.no_grad()
    def generate(self, batch, n_return_sequences=1, return_codes=False):
        """
        Generate recommendations using iterative denoising
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
        steps_used = 0
        for t in reversed(range(1, self.T + 1)):
            steps_used += 1
            
            # Construct input with current codes
            input_tokens = self.item_id2tokens[batch['input_ids']]
            input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
            
            # Item position embedding removed - keep it simple like RPG
            
            # Get embeddings for current codes
            target_embs = self.gpt2.wte(current_codes).mean(dim=1, keepdim=True)
            
            # Time embedding removed - not needed without proper diffusion training
            
            # Concatenate
            all_embs = torch.cat([input_embs, target_embs], dim=1)
            
            # Extend attention mask
            attention_mask_extended = torch.cat([
                batch['attention_mask'],
                torch.ones(batch_size, 1, device=device, dtype=batch['attention_mask'].dtype)
            ], dim=1)
            
            # Forward
            outputs = self.gpt2(inputs_embeds=all_embs, attention_mask=attention_mask_extended)
            target_hidden = outputs.last_hidden_state[:, -1, :]
            
            # Predict codes
            final_states = torch.cat([
                self.pred_heads[i](target_hidden).unsqueeze(1) 
                for i in range(self.n_pred_head)
            ], dim=1)
            
            # Get predictions for each digit
            final_states_norm = F.normalize(final_states, dim=-1)
            token_emb_norm = F.normalize(self.gpt2.wte.weight[1:1+self.n_pred_head*self.config['codebook_size']], dim=-1)
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
            
            predicted_codes = torch.stack(predicted_codes, dim=1)
            confidence_scores = torch.stack(confidence_scores, dim=1)
            
            # Update strategy: determine fixed number of codes per step
            if t > 1:
                if self.codes_per_step is not None:
                    # Fixed number of codes to determine each step per sample
                    num_to_determine = self.codes_per_step
                else:
                    # Use mask ratio: higher mask ratio -> determine fewer codes
                    mask_ratio = self.get_mask_ratio(t - 1)
                    num_to_determine = int(self.n_pred_head * (1 - mask_ratio))
                
                # Find positions that are still MASK
                mask_positions = (current_codes == self.mask_token_id)
                
                if mask_positions.any():
                    # Get confidence scores only for MASK positions
                    mask_confidence = confidence_scores.clone()
                    mask_confidence[~mask_positions] = -float('inf')  # Set non-MASK positions to -inf
                    
                    # Keep top-k confident predictions among MASK positions
                    _, top_k_indices = mask_confidence.topk(num_to_determine, dim=1)
                    
                    # Update current_codes (only change positions that are still MASK)
                    updated_any = False
                    mask_count_before = (current_codes == self.mask_token_id).sum().item()
                    for b in range(batch_size):
                        for idx in top_k_indices[b]:
                            if current_codes[b, idx] == self.mask_token_id:
                                current_codes[b, idx] = predicted_codes[b, idx]
                                updated_any = True
                    mask_count_after = (current_codes == self.mask_token_id).sum().item()
                    # print(f"[DEBUG] Step {steps_used}, t={t}: determined {num_to_determine} codes per sample, total masks {mask_count_before} -> {mask_count_after}, updated_any={updated_any}")
                    
                    # Early termination: if no more MASK tokens, break
                    if not updated_any:
                        # print(f"[DEBUG] Early termination at step {steps_used}/{self.T}, t={t}, no updates")
                        break
                    
                    # Also check if all codes are determined
                    if (current_codes != self.mask_token_id).all():
                        # print(f"[DEBUG] Early termination at step {steps_used}/{self.T}, t={t}, all codes determined")
                        break
                else:
                    # print(f"[DEBUG] Step {steps_used}, t={t}: no MASK positions left, breaking")
                    break
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
        """Convert semantic codes to item logits"""
        batch_size = codes.shape[0]
        all_item_codes = self.item_id2tokens[1:]  # (n_items-1, n_digit)
        
        if method == 'count':
            # Method 1: Count matching codes
            item_logits = torch.zeros(batch_size, self.dataset.n_items - 1, device=codes.device)
            for b in range(batch_size):
                matches = (codes[b].unsqueeze(0) == all_item_codes).sum(dim=1)
                item_logits[b] = matches.float()
        
        elif method == 'embedding':
            # Method 2: Use embedding similarity
            code_embs = self.gpt2.wte(codes).mean(dim=1)
            all_item_embs = self.gpt2.wte(all_item_codes).mean(dim=1)
            
            code_embs_norm = F.normalize(code_embs, dim=-1)
            all_item_embs_norm = F.normalize(all_item_embs, dim=-1)
            item_logits = torch.matmul(code_embs_norm, all_item_embs_norm.T)
        
        return item_logits

    def check_code_validity(self, codes: torch.Tensor) -> dict:
        """Check if generated codes match any real items"""
        batch_size = codes.shape[0]
        all_item_codes = self.item_id2tokens[1:]  # (n_items-1, n_digit)
        
        exact_matches = 0
        max_matches = []
        
        for b in range(batch_size):
            matches = (codes[b].unsqueeze(0) == all_item_codes).all(dim=1)
            
            if matches.any():
                exact_matches += 1
            
            num_matching_codes = (codes[b].unsqueeze(0) == all_item_codes).sum(dim=1)
            max_matches.append(num_matching_codes.max().item())
        
        return {
            'exact_match_rate': exact_matches / batch_size,
            'avg_max_matching_codes': sum(max_matches) / len(max_matches),
            'exact_matches': exact_matches,
            'total': batch_size
        }
