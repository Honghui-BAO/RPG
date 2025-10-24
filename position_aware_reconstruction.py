"""
位置感知的Reconstruction Loss实现方案
"""

import torch
import torch.nn.functional as F

class PositionAwareReconstructionLoss:
    """位置感知的重构损失计算"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.position_weights = self._init_position_weights()
    
    def _init_position_weights(self):
        """初始化位置权重 - 不同位置有不同的重要性"""
        # 方案1: 线性权重 - 后面的位置权重更高
        max_len = self.config.get('max_item_seq_len', 50)
        weights = torch.linspace(0.5, 1.5, max_len)  # 位置越靠后权重越高
        return weights
    
    def calculate_position_aware_loss(self, masked_final_states, mask_positions, original_tokens):
        """计算位置感知的重构损失"""
        batch_size, seq_len = mask_positions.shape
        device = mask_positions.device
        
        # 获取位置信息
        position_indices = torch.nonzero(mask_positions, as_tuple=False)  # (num_masked, 2) [batch_idx, seq_idx]
        batch_indices = position_indices[:, 0]  # (num_masked,)
        seq_indices = position_indices[:, 1]    # (num_masked,)
        
        # 获取位置权重
        position_weights = self.position_weights[seq_indices].to(device)  # (num_masked,)
        
        # 获取被mask位置的states
        masked_states = masked_final_states[mask_positions]  # (num_masked, n_pred_head, n_embd)
        original_tokens_masked = original_tokens[mask_positions]  # (num_masked, n_digit)
        
        # 计算每个位置的损失
        position_losses = []
        for i in range(len(batch_indices)):
            # 获取该位置的权重
            pos_weight = position_weights[i]
            
            # 计算该位置的重构损失
            pos_loss = self._calculate_single_position_loss(
                masked_states[i], original_tokens_masked[i]
            )
            
            # 应用位置权重
            weighted_loss = pos_loss * pos_weight
            position_losses.append(weighted_loss)
        
        return torch.mean(torch.stack(position_losses))
    
    def _calculate_single_position_loss(self, masked_state, original_token):
        """计算单个位置的重构损失"""
        # masked_state: (n_pred_head, n_embd)
        # original_token: (n_digit,)
        
        masked_state = F.normalize(masked_state, dim=-1)
        masked_state = torch.chunk(masked_state, self.model.n_pred_head, dim=0)
        
        # 获取token embeddings
        token_emb = self.model.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.model.n_pred_head, dim=0)
        
        # 计算每个digit的损失
        digit_losses = []
        for i in range(self.model.n_pred_head):
            digit_logits = torch.matmul(masked_state[i], token_embs[i].T) / self.model.temperature
            digit_labels = original_token[i] - i * self.model.config['codebook_size'] - 1
            digit_loss = self.model.loss_fct(digit_logits, digit_labels.unsqueeze(0))
            digit_losses.append(digit_loss)
        
        return torch.mean(torch.stack(digit_losses))


class ContextAwareReconstructionLoss:
    """上下文感知的重构损失计算"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def calculate_context_aware_loss(self, masked_final_states, mask_positions, original_tokens, attention_mask):
        """计算上下文感知的重构损失"""
        batch_size, seq_len = mask_positions.shape
        
        # 为每个被mask的位置计算上下文信息
        context_losses = []
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if mask_positions[batch_idx, seq_idx]:
                    # 计算该位置的上下文损失
                    context_loss = self._calculate_context_loss(
                        batch_idx, seq_idx, 
                        masked_final_states, 
                        original_tokens, 
                        attention_mask
                    )
                    context_losses.append(context_loss)
        
        return torch.mean(torch.stack(context_losses)) if context_losses else torch.tensor(0.0)
    
    def _calculate_context_loss(self, batch_idx, seq_idx, masked_final_states, original_tokens, attention_mask):
        """计算单个位置的上下文损失"""
        # 获取该位置的预测状态
        masked_state = masked_final_states[batch_idx, seq_idx]  # (n_pred_head, n_embd)
        original_token = original_tokens[batch_idx, seq_idx]     # (n_digit,)
        
        # 获取上下文信息（前后位置的states）
        context_states = self._get_context_states(
            batch_idx, seq_idx, masked_final_states, attention_mask
        )
        
        # 结合上下文信息进行预测
        enhanced_state = self._enhance_with_context(masked_state, context_states)
        
        # 计算重构损失
        return self._calculate_single_position_loss(enhanced_state, original_token)
    
    def _get_context_states(self, batch_idx, seq_idx, masked_final_states, attention_mask):
        """获取上下文状态"""
        seq_len = masked_final_states.shape[1]
        
        # 获取前后位置的states
        prev_states = []
        next_states = []
        
        # 前向上下文
        for i in range(max(0, seq_idx - 2), seq_idx):
            if attention_mask[batch_idx, i]:
                prev_states.append(masked_final_states[batch_idx, i])
        
        # 后向上下文
        for i in range(seq_idx + 1, min(seq_len, seq_idx + 3)):
            if attention_mask[batch_idx, i]:
                next_states.append(masked_final_states[batch_idx, i])
        
        return {
            'prev_states': prev_states,
            'next_states': next_states
        }
    
    def _enhance_with_context(self, masked_state, context_states):
        """使用上下文信息增强预测状态"""
        # 简单的上下文融合策略
        enhanced_state = masked_state.clone()
        
        if context_states['prev_states']:
            prev_avg = torch.mean(torch.stack(context_states['prev_states']), dim=0)
            enhanced_state = enhanced_state + 0.3 * prev_avg
        
        if context_states['next_states']:
            next_avg = torch.mean(torch.stack(context_states['next_states']), dim=0)
            enhanced_state = enhanced_state + 0.3 * next_avg
        
        return enhanced_state


class AdaptiveReconstructionLoss:
    """自适应重构损失计算"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.difficulty_estimator = self._init_difficulty_estimator()
    
    def _init_difficulty_estimator(self):
        """初始化难度估计器"""
        # 基于item频率的难度估计
        return None  # 需要根据具体数据集实现
    
    def calculate_adaptive_loss(self, masked_final_states, mask_positions, original_tokens):
        """计算自适应重构损失"""
        # 根据item的难度调整损失权重
        # 稀有item的损失权重更高
        # 常见item的损失权重较低
        
        batch_size, seq_len = mask_positions.shape
        adaptive_losses = []
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if mask_positions[batch_idx, seq_idx]:
                    # 估计该item的预测难度
                    difficulty = self._estimate_difficulty(original_tokens[batch_idx, seq_idx])
                    
                    # 计算基础损失
                    base_loss = self._calculate_single_position_loss(
                        masked_final_states[batch_idx, seq_idx],
                        original_tokens[batch_idx, seq_idx]
                    )
                    
                    # 应用难度权重
                    adaptive_loss = base_loss * (1.0 + difficulty)
                    adaptive_losses.append(adaptive_loss)
        
        return torch.mean(torch.stack(adaptive_losses)) if adaptive_losses else torch.tensor(0.0)
    
    def _estimate_difficulty(self, item_tokens):
        """估计item的预测难度"""
        # 简单的难度估计：基于token的多样性
        unique_tokens = torch.unique(item_tokens)
        diversity = len(unique_tokens) / len(item_tokens)
        return diversity  # 多样性越高，难度越大
