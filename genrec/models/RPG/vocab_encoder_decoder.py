# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel


class EncoderMLP(nn.Module):
    """
    Encoder MLP that maps from 192d to 768d.
    
    Args:
        input_dim (int): Input dimension (default: 192)
        output_dim (int): Output dimension (default: 768)
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of hidden layers
        dropout (float): Dropout rate
    """
    def __init__(self, input_dim=192, output_dim=768, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim)
        """
        return self.mlp(x)


class DecoderMLP(nn.Module):
    """
    Decoder MLP that maps from 768d to 768d (for direct summation).
    
    Args:
        input_dim (int): Input dimension (default: 768)
        output_dim (int): Output dimension (default: 768)
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of hidden layers
        dropout (float): Dropout rate
    """
    def __init__(self, input_dim=768, output_dim=768, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim)
        """
        return self.mlp(x)


class VocabTokenizer(nn.Module):
    """
    Vocabulary-based tokenizer that:
    1. Splits 768d embeddings into 4 parts (192d each)
    2. Encodes each part to 768d
    3. Finds closest token from T5 word embeddings
    4. Decodes token embeddings to 768d
    5. Sums 4 decoded embeddings and computes reconstruction loss
    
    Args:
        config (dict): Configuration dictionary with keys:
            - vocab_hidden_dim: Hidden dimension for MLPs
            - vocab_num_layers: Number of layers in MLPs
            - vocab_dropout: Dropout rate
            - device: Device to use for computation
        t5_word_embeddings (torch.Tensor): T5 word embedding matrix of shape (vocab_size, 768)
        log_fn (callable): Logging function
    """
    def __init__(self, config, t5_word_embeddings, log_fn=print):
        super().__init__()
        self.config = config
        self.log = log_fn
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store T5 word embeddings (vocab_size, 768)
        self.register_buffer('t5_word_embeddings', t5_word_embeddings)
        self.vocab_size = t5_word_embeddings.shape[0]
        self.embedding_dim = t5_word_embeddings.shape[1]  # Should be 768
        
        # Split dimensions
        self.n_splits = 4
        self.split_dim = self.embedding_dim // self.n_splits  # 192
        
        # Create 4 encoder MLPs (192d -> 768d)
        self.encoders = nn.ModuleList([
            EncoderMLP(
                input_dim=self.split_dim,
                output_dim=self.embedding_dim,
                hidden_dim=config.get('vocab_hidden_dim', 512),
                num_layers=config.get('vocab_num_layers', 2),
                dropout=config.get('vocab_dropout', 0.1)
            ) for _ in range(self.n_splits)
        ])
        
        # Create 4 decoder MLPs (768d -> 768d)
        self.decoders = nn.ModuleList([
            DecoderMLP(
                input_dim=self.embedding_dim,
                output_dim=self.embedding_dim,
                hidden_dim=config.get('vocab_hidden_dim', 512),
                num_layers=config.get('vocab_num_layers', 2),
                dropout=config.get('vocab_dropout', 0.1)
            ) for _ in range(self.n_splits)
        ])
        
        self.to(self.device)
    
    def split_embedding(self, embedding):
        """
        Split embedding into 4 equal parts.
        
        Args:
            embedding (torch.Tensor): Input embedding of shape (..., 768)
        
        Returns:
            list of torch.Tensor: 4 tensors each of shape (..., 192)
        """
        return torch.chunk(embedding, self.n_splits, dim=-1)
    
    def find_nearest_tokens(self, encoded_parts):
        """
        Find nearest tokens from T5 vocabulary for each encoded part.
        
        Args:
            encoded_parts (list of torch.Tensor): 4 tensors each of shape (batch_size, 768)
        
        Returns:
            torch.Tensor: Token indices of shape (batch_size, 4)
            list of torch.Tensor: Retrieved token embeddings (4 tensors of shape (batch_size, 768))
        """
        batch_size = encoded_parts[0].shape[0]
        token_indices = []
        token_embeddings = []
        
        # Normalize T5 word embeddings once
        normalized_t5_embs = F.normalize(self.t5_word_embeddings, dim=-1)  # (vocab_size, 768)
        
        for i, encoded_part in enumerate(encoded_parts):
            # Normalize encoded part
            normalized_encoded = F.normalize(encoded_part, dim=-1)  # (batch_size, 768)
            
            # Compute cosine similarity with all T5 word embeddings
            similarities = torch.matmul(normalized_encoded, normalized_t5_embs.T)  # (batch_size, vocab_size)
            
            # Find nearest token
            nearest_token_idx = torch.argmax(similarities, dim=-1)  # (batch_size,)
            token_indices.append(nearest_token_idx)
            
            # Retrieve token embeddings
            token_emb = self.t5_word_embeddings[nearest_token_idx]  # (batch_size, 768)
            token_embeddings.append(token_emb)
        
        # Stack token indices
        token_indices = torch.stack(token_indices, dim=-1)  # (batch_size, 4)
        
        return token_indices, token_embeddings
    
    def decode_token_embeddings(self, token_embeddings):
        """
        Decode token embeddings using decoder MLPs.
        
        Args:
            token_embeddings (list of torch.Tensor): 4 tensors each of shape (batch_size, 768)
        
        Returns:
            list of torch.Tensor: 4 decoded tensors each of shape (batch_size, 768)
        """
        decoded_parts = []
        for i, token_emb in enumerate(token_embeddings):
            decoded = self.decoders[i](token_emb)  # (batch_size, 768)
            decoded_parts.append(decoded)
        return decoded_parts
    
    def forward(self, embeddings, return_tokens=False):
        """
        Forward pass: encode -> find tokens -> decode -> reconstruct.
        
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, 768)
            return_tokens (bool): Whether to return token indices
        
        Returns:
            dict: Dictionary containing:
                - reconstructed: Reconstructed embeddings (batch_size, 768)
                - loss: Reconstruction loss (scalar)
                - tokens (optional): Token indices (batch_size, 4)
        """
        batch_size = embeddings.shape[0]
        
        # 1. Split embeddings into 4 parts
        split_embs = self.split_embedding(embeddings)  # 4 x (batch_size, 192)
        
        # 2. Encode each part to 768d
        encoded_parts = []
        for i, split_emb in enumerate(split_embs):
            encoded = self.encoders[i](split_emb)  # (batch_size, 768)
            encoded_parts.append(encoded)
        
        # 3. Find nearest tokens from T5 vocabulary
        token_indices, token_embeddings = self.find_nearest_tokens(encoded_parts)
        
        # 4. Decode token embeddings
        decoded_parts = self.decode_token_embeddings(token_embeddings)
        
        # 5. Sum decoded parts to get reconstructed embedding
        reconstructed = torch.stack(decoded_parts, dim=0).sum(dim=0)  # (batch_size, 768)
        
        # 6. Compute reconstruction loss (MSE)
        loss = F.mse_loss(reconstructed, embeddings)
        
        result = {
            'reconstructed': reconstructed,
            'loss': loss
        }
        
        if return_tokens:
            result['tokens'] = token_indices
        
        return result
    
    def tokenize_items(self, embeddings):
        """
        Tokenize items into 4-token representations.
        
        Args:
            embeddings (torch.Tensor): Item embeddings of shape (n_items, 768)
        
        Returns:
            torch.Tensor: Token indices of shape (n_items, 4)
        """
        with torch.no_grad():
            # Split embeddings
            split_embs = self.split_embedding(embeddings)
            
            # Encode each part
            encoded_parts = []
            for i, split_emb in enumerate(split_embs):
                encoded = self.encoders[i](split_emb)
                encoded_parts.append(encoded)
            
            # Find nearest tokens
            token_indices, _ = self.find_nearest_tokens(encoded_parts)
            
            return token_indices
    
    def train_tokenizer(self, embeddings, optimizer, n_epochs, batch_size=256):
        """
        Train the vocab tokenizer.
        
        Args:
            embeddings (torch.Tensor): Training embeddings of shape (n_items, 768)
            optimizer (torch.optim.Optimizer): Optimizer
            n_epochs (int): Number of training epochs
            batch_size (int): Batch size
        
        Returns:
            list: Training losses per epoch
        """
        self.train()
        n_items = embeddings.shape[0]
        losses = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Shuffle indices
            perm = torch.randperm(n_items)
            
            # Mini-batch training
            for i in range(0, n_items, batch_size):
                batch_indices = perm[i:i+batch_size]
                batch_embs = embeddings[batch_indices].to(self.device)
                
                # Forward pass
                output = self.forward(batch_embs)
                loss = output['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            self.log(f'[VOCAB TOKENIZER] Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}')
        
        return losses
