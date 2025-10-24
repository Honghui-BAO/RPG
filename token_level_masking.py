"""
Token-Level Masking实现方案
"""

import torch
import torch.nn.functional as F

class TokenLevelMasking:
    """Token级别的masking策略"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.mask_ratio = config.get('mask_ratio', 0.15)
        self.mask_token_id = 0
    
    def create_token_level_masked_tokens(self, input_tokens, attention_mask):
        """
        创建token级别的masked tokens
        
        Args:
            input_tokens: (batch_size, seq_len, n_digit)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            masked_tokens: (batch_size, seq_len, n_digit)
            mask_positions: (batch_size, seq_len, n_digit)
            original_tokens: (batch_size, seq_len, n_digit)
        """
        batch_size, seq_len, n_digit = input_tokens.shape
        device = input_tokens.device
        
        # 创建token级别的mask位置
        mask_positions = torch.zeros_like(input_tokens, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            valid_len = attention_mask[i].sum().item()
            if valid_len > 0:
                # 计算需要mask的token总数
                total_tokens = valid_len * n_digit
                num_to_mask = max(1, int(total_tokens * self.mask_ratio))
                
                # 随机选择token位置进行mask
                valid_tokens = []
                for seq_idx in range(valid_len):
                    for digit_idx in range(n_digit):
                        valid_tokens.append((seq_idx, digit_idx))
                
                # 随机选择要mask的token
                selected_tokens = torch.randperm(len(valid_tokens), device=device)[:num_to_mask]
                
                for token_idx in selected_tokens:
                    seq_idx, digit_idx = valid_tokens[token_idx]
                    mask_positions[i, seq_idx, digit_idx] = True
        
        # 创建masked tokens
        masked_tokens = input_tokens.clone()
        original_tokens = input_tokens.clone()
        masked_tokens[mask_positions] = self.mask_token_id
        
        return masked_tokens, mask_positions, original_tokens
    
    def calculate_token_level_loss(self, masked_final_states, mask_positions, original_tokens):
        """计算token级别的重构损失"""
        # 只对被mask的token计算损失
        masked_tokens = original_tokens[mask_positions]  # (num_masked_tokens,)
        
        if len(masked_tokens) == 0:
            return torch.tensor(0.0, device=original_tokens.device)
        
        # 获取对应的states
        batch_size, seq_len, n_digit = mask_positions.shape
        masked_states = []
        
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(n_digit):
                    if mask_positions[i, j, k]:
                        # 获取该位置的state
                        state = masked_final_states[i, j]  # (n_pred_head, n_embd)
                        masked_states.append(state[k])  # 对应digit k的state
        
        masked_states = torch.stack(masked_states)  # (num_masked_tokens, n_embd)
        
        # 计算重构损失
        # 这里需要根据具体的token embedding计算
        # 简化实现
        return torch.mean(torch.norm(masked_states, dim=-1))


class HybridMasking:
    """混合masking策略：结合item-level和token-level"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.item_mask_ratio = config.get('item_mask_ratio', 0.1)
        self.token_mask_ratio = config.get('token_mask_ratio', 0.05)
    
    def create_hybrid_masked_tokens(self, input_tokens, attention_mask):
        """创建混合masked tokens"""
        batch_size, seq_len, n_digit = input_tokens.shape
        device = input_tokens.device
        
        # 1. Item-level masking
        item_mask_positions = torch.zeros_like(attention_mask, dtype=torch.bool, device=device)
        for i in range(batch_size):
            valid_len = attention_mask[i].sum().item()
            if valid_len > 0:
                num_items_to_mask = max(1, int(valid_len * self.item_mask_ratio))
                valid_positions = torch.arange(valid_len, device=device)
                masked_items = valid_positions[torch.randperm(valid_len, device=device)[:num_items_to_mask]]
                item_mask_positions[i, masked_items] = True
        
        # 2. Token-level masking (在未被item-level mask的位置)
        token_mask_positions = torch.zeros_like(input_tokens, dtype=torch.bool, device=device)
        for i in range(batch_size):
            for j in range(seq_len):
                if not item_mask_positions[i, j] and attention_mask[i, j]:
                    # 在这个item内部进行token-level masking
                    num_tokens_to_mask = max(1, int(n_digit * self.token_mask_ratio))
                    token_indices = torch.randperm(n_digit, device=device)[:num_tokens_to_mask]
                    token_mask_positions[i, j, token_indices] = True
        
        # 3. 合并masking
        item_mask_expanded = item_mask_positions.unsqueeze(-1).expand(-1, -1, n_digit)
        final_mask_positions = item_mask_expanded | token_mask_positions
        
        # 创建masked tokens
        masked_tokens = input_tokens.clone()
        original_tokens = input_tokens.clone()
        masked_tokens[final_mask_positions] = 0
        
        return masked_tokens, final_mask_positions, original_tokens


class AdaptiveMasking:
    """自适应masking策略：根据item特征调整masking策略"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.item_frequency = self._init_item_frequency()
    
    def _init_item_frequency(self):
        """初始化item频率统计"""
        # 这里需要根据实际数据集计算
        return {}
    
    def create_adaptive_masked_tokens(self, input_tokens, attention_mask):
        """创建自适应masked tokens"""
        batch_size, seq_len, n_digit = input_tokens.shape
        device = input_tokens.device
        
        mask_positions = torch.zeros_like(input_tokens, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask[i, j]:
                    # 根据item特征决定masking策略
                    item_tokens = input_tokens[i, j]
                    masking_strategy = self._choose_masking_strategy(item_tokens)
                    
                    if masking_strategy == 'item_level':
                        # 整个item都mask
                        mask_positions[i, j, :] = True
                    elif masking_strategy == 'token_level':
                        # 部分token mask
                        num_to_mask = max(1, int(n_digit * 0.3))
                        token_indices = torch.randperm(n_digit, device=device)[:num_to_mask]
                        mask_positions[i, j, token_indices] = True
                    elif masking_strategy == 'selective':
                        # 选择性mask重要token
                        important_tokens = self._identify_important_tokens(item_tokens)
                        mask_positions[i, j, important_tokens] = True
        
        # 创建masked tokens
        masked_tokens = input_tokens.clone()
        original_tokens = input_tokens.clone()
        masked_tokens[mask_positions] = 0
        
        return masked_tokens, mask_positions, original_tokens
    
    def _choose_masking_strategy(self, item_tokens):
        """根据item特征选择masking策略"""
        # 简化实现：基于token的多样性
        unique_tokens = torch.unique(item_tokens)
        diversity = len(unique_tokens) / len(item_tokens)
        
        if diversity > 0.8:
            return 'item_level'  # 高多样性，整个item mask
        elif diversity > 0.5:
            return 'token_level'  # 中等多样性，部分token mask
        else:
            return 'selective'  # 低多样性，选择性mask
    
    def _identify_important_tokens(self, item_tokens):
        """识别重要的token"""
        # 简化实现：选择值较大的token
        _, important_indices = torch.topk(item_tokens, k=len(item_tokens)//2)
        return important_indices
