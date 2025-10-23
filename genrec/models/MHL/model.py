import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
import random


class MHL(AbstractModel):
    """
    Masked Hierarchical Learning (MHL) Model
    
    Based on RPG but with masking during training and reconstruction loss.
    
    Key differences from RPG:
    1. Training time masking: Randomly mask some tokens in the input sequence
    2. Reconstruction loss: Predict masked tokens to reconstruct original sequence
    3. Maintains item-level encoding like RPG for efficiency
    
    Architecture:
    - Item-level encoding: Average pool tokens to get item representations
    - GPT2 backbone with causal attention
    - 32 prediction heads for token prediction
    - Masking strategy: Random mask ratio during training
    """
    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(MHL, self).__init__(config, dataset, tokenizer)
        
        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        
        # GPT2 backbone with causal attention
        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=config['max_item_seq_len'],
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
        )
        self.gpt2 = GPT2Model(gpt2config)
        
        # Prediction heads for each codebook position
        self.n_pred_head = self.tokenizer.n_digit
        self.pred_heads = nn.ModuleList([
            nn.Linear(config['n_embd'], config['n_embd']) 
            for _ in range(self.n_pred_head)
        ])
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Temperature for logits
        self.temperature = self.config['temperature']
        
        # Masking parameters
        self.mask_ratio = config.get('mask_ratio', 0.15)  # Default 15% masking ratio
        self.mask_token_id = tokenizer.mask_token_id
        
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
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def apply_masking(self, input_tokens, mask_ratio=None):
        """
        Apply random masking to input tokens during training
        
        Args:
            input_tokens: (batch_size, seq_len, n_codebook) - input token sequences
            mask_ratio: ratio of tokens to mask (if None, use self.mask_ratio)
            
        Returns:
            masked_tokens: tokens with some positions masked
            mask: boolean mask indicating which positions were masked
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        batch_size, seq_len, n_codebook = input_tokens.shape
        device = input_tokens.device
        
        # Create mask for each position
        mask = torch.rand(batch_size, seq_len, n_codebook, device=device) < mask_ratio
        
        # Create masked tokens
        masked_tokens = input_tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        return masked_tokens, mask

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        # Item-level encoding: aggregate tokens to get item representations
        input_tokens = self.item_id2tokens[batch['input_ids']]  # (batch_size, seq_len, n_codebook)
        batch_size, seq_len, n_codebook = input_tokens.shape
        
        # Debug: check validity
        if (input_tokens < 0).any() or (input_tokens >= self.gpt2.config.vocab_size).any():
            print(f"[ERROR] Invalid tokens in MHL!")
            print(f"  vocab_size: {self.gpt2.config.vocab_size}")
            print(f"  input_tokens range: [{input_tokens.min()}, {input_tokens.max()}]")
        
        # Apply masking during training
        if self.training:
            masked_tokens, mask = self.apply_masking(input_tokens)
        else:
            masked_tokens = input_tokens
            mask = torch.zeros_like(input_tokens, dtype=torch.bool)
        
        # Get token embeddings and aggregate to item-level
        input_embs = self.gpt2.wte(masked_tokens)  # (batch_size, seq_len, n_codebook, n_embd)
        # Average pool across n_codebook dimension to get item-level embeddings
        input_embs = input_embs.mean(dim=-2)  # (batch_size, seq_len, n_embd)
        
        # Debug: check sequence length
        if seq_len > self.gpt2.config.n_positions:
            print(f"[WARNING] MHL: Sequence length {seq_len} exceeds GPT2 limit {self.gpt2.config.n_positions}!")
        
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        
        # Apply prediction heads to the last item's embedding (follow main branch RPG exactly)
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        
        if return_loss:
            # Temporarily disable reconstruction loss - only use next-item prediction loss
            recon_loss = 0.0
            
            # Compute next-item prediction loss (same as RPG)
            assert 'labels' in batch, 'The batch must contain the labels.'
            
            label_mask = batch['labels'].view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            losses = [
                self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1)
                for i in range(self.n_pred_head)
            ]
            next_item_loss = torch.mean(torch.stack(losses))
            
            # Combine losses (only next-item loss for now)
            outputs.loss = next_item_loss
            
        return outputs

    def compute_reconstruction_loss(self, sequence_hidden_states, original_tokens, mask, labels):
        """
        Compute reconstruction loss for masked tokens
        
        Args:
            sequence_hidden_states: (batch_size, seq_len, n_embd) - sequence hidden states from GPT2
            original_tokens: (batch_size, seq_len, n_codebook) - original tokens
            mask: (batch_size, seq_len, n_codebook) - mask indicating masked positions
            labels: target labels for next-item prediction
            
        Returns:
            loss: combined reconstruction and next-item prediction loss
        """
        batch_size, seq_len, n_embd = sequence_hidden_states.shape
        
        # 1. Reconstruction loss for masked tokens
        recon_loss = 0.0
        if mask.any():
            # Get token embeddings for prediction
            # Exclude PAD (0), EOS, and MASK tokens - only use semantic tokens
            token_emb = self.gpt2.wte.weight[1:self.eos_token]  # Only semantic tokens
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            
            # Normalize sequence hidden states
            sequence_hidden_norm = F.normalize(sequence_hidden_states, dim=-1)
            
            # For each codebook position, compute prediction loss
            for i in range(self.n_pred_head):
                # Get predictions for this codebook position at each sequence position
                # sequence_hidden_norm: (batch_size, seq_len, n_embd)
                # token_embs[i]: (vocab_size, n_embd)
                pred_logits = torch.matmul(sequence_hidden_norm, token_embs[i].T) / self.temperature
                # pred_logits: (batch_size, seq_len, vocab_size)
                
                # Get original tokens for this position
                original_tokens_i = original_tokens[:, :, i]  # (batch_size, seq_len)
                mask_i = mask[:, :, i]  # (batch_size, seq_len)
                
                # Only compute loss for masked positions
                if mask_i.any():
                    # Flatten for loss computation
                    pred_logits_flat = pred_logits.reshape(-1, token_embs[i].shape[0])
                    original_tokens_flat = original_tokens_i.view(-1)
                    mask_flat = mask_i.view(-1)
                    
                    # Select masked positions
                    masked_pred = pred_logits_flat[mask_flat]
                    masked_target = original_tokens_flat[mask_flat]
                    
                    # Compute loss for this codebook position
                    if len(masked_target) > 0:
                        # Adjust target indices for this codebook
                        adjusted_target = masked_target - i * self.config['codebook_size'] - 1
                        
                        # Debug: check target indices validity
                        if (adjusted_target < 0).any() or (adjusted_target >= token_embs[i].shape[0]).any():
                            print(f"[ERROR] Invalid target indices in MHL reconstruction loss!")
                            print(f"  Codebook {i}: original_target range: [{masked_target.min()}, {masked_target.max()}]")
                            print(f"  Codebook {i}: adjusted_target range: [{adjusted_target.min()}, {adjusted_target.max()}]")
                            print(f"  Codebook {i}: vocab_size: {token_embs[i].shape[0]}")
                            print(f"  Codebook {i}: codebook_size: {self.config['codebook_size']}")
                            # Skip this batch to avoid crash
                            continue
                        
                        recon_loss += self.loss_fct(masked_pred, adjusted_target)
        
        # 2. Next-item prediction loss (same as RPG)
        # Note: This will be computed in the main forward method using final_states
        # For now, we only return the reconstruction loss
        return recon_loss

    def generate(self, batch, n_return_sequences=1):
        """Generate predictions (same as RPG)"""
        outputs = self.forward(batch, return_loss=False)
        # final_states is already (batch_size, n_pred_head, n_embd) from the last token
        states = outputs.final_states  # (batch_size, n_pred_head, n_embd)
        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [torch.matmul(states[:,i,:], token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            return self.graph_propagation(batch, token_logits, n_return_sequences)
        else:
            generated_tokens = torch.argmax(token_logits, dim=-1)
            return generated_tokens.unsqueeze(-1), None
