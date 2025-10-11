# RPG → LLaDA 改编设计文档

参考论文：[Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992)

## 1. 核心改变对比

### 当前RPG（Autoregressive）
```
用户序列: [item_1, item_2, item_3]
         ↓ 每个item → 32个codes
GPT2输入: [codes_1, codes_2, codes_3]
         ↓ 自回归预测
输出: 下一个item的32个codes
Loss: CrossEntropy(predicted, target)
```

### LLaDA风格（Diffusion）
```
用户序列: [item_1, item_2, item_3] + target_item
         ↓ 所有item → 32个codes
         ↓ Forward: 逐步mask target_item的codes
时刻t: [codes_1, codes_2, codes_3, masked_target]
         ↓ Transformer预测
         ↓ Reverse: 预测被mask的codes
输出: 恢复的target codes
Loss: 预测被mask位置的codes
```

## 2. Forward Process (加噪/Masking)

### 方案A：离散扩散（推荐用于RPG）

```python
# Target item的32个codes
target_codes = [c_0, c_1, c_2, ..., c_31]  # 每个c_i ∈ [0, 255]

# Forward process: 在时间步t，mask一定比例的codes
t ~ Uniform(1, T)  # T=32 或更大
mask_ratio = t / T

# 随机选择mask_ratio比例的位置
num_masked = int(32 * mask_ratio)
masked_positions = random.sample(range(32), num_masked)

# 将这些位置替换为特殊token [MASK]
masked_codes = target_codes.copy()
for pos in masked_positions:
    masked_codes[pos] = MASK_TOKEN_ID  # 例如 8194

# 输入到模型
input_sequence = [item_1_codes, item_2_codes, item_3_codes, masked_codes]
```

### Forward Process的关键参数
- **T**: 总扩散步数（建议T=32，对应32个codes）
- **Masking schedule**: 
  - Linear: mask_ratio = t/T
  - Cosine: mask_ratio = cos(π*t/2T)
  - Square: mask_ratio = (t/T)^2

## 3. Reverse Process (去噪/Denoising)

### 模型架构修改

```python
class LLaDARecommender(AbstractModel):
    def __init__(self, config, dataset, tokenizer):
        super().__init__(config, dataset, tokenizer)
        
        # 添加时间步embedding
        self.time_embed = nn.Embedding(config['diffusion_steps'] + 1, config['n_embd'])
        
        # 添加MASK token
        self.mask_token_id = tokenizer.vocab_size  # 8194
        
        # GPT2 Transformer (保持不变)
        self.gpt2 = GPT2Model(gpt2config)
        
        # Prediction heads (保持32个)
        self.pred_heads = nn.Sequential(*[ResBlock(config['n_embd']) for _ in range(32)])
```

### 训练过程

```python
def forward(self, batch, return_loss=True):
    # 1. 获取target item的codes
    target_codes = self.item_id2tokens[batch['labels']]  # (batch, 32)
    
    # 2. 采样时间步
    t = torch.randint(1, self.T + 1, (batch_size,))  # (batch,)
    
    # 3. Forward process: mask codes
    masked_codes, mask_positions = self.forward_diffusion(target_codes, t)
    
    # 4. 构建输入序列
    input_ids = torch.cat([
        batch['input_ids'],  # 用户历史
        masked_codes.unsqueeze(1)  # 被mask的target
    ], dim=1)
    
    # 5. 添加时间步信息
    time_emb = self.time_embed(t)  # (batch, n_embd)
    
    # 6. 通过Transformer
    input_tokens = self.item_id2tokens[input_ids]  # (batch, seq_len, 32)
    input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)  # (batch, seq_len, n_embd)
    
    # 注入时间信息（方式1：加到最后一个位置）
    input_embs[:, -1, :] += time_emb
    
    outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=batch['attention_mask'])
    
    # 7. 预测被mask的codes
    final_states = self.pred_heads(outputs.last_hidden_state[:, -1, :])  # (batch, 32, n_embd)
    
    # 8. 计算loss（只在被mask的位置）
    if return_loss:
        losses = []
        for i in range(32):
            if mask_positions[i]:  # 只计算被mask位置的loss
                logits = torch.matmul(
                    F.normalize(final_states[:, i, :], dim=-1),
                    F.normalize(self.gpt2.wte.weight[1:-1], dim=-1).T
                )
                losses.append(F.cross_entropy(logits, target_codes[:, i]))
        
        outputs.loss = torch.mean(torch.stack(losses))
    
    return outputs
```

## 4. Inference (采样过程)

### 从噪声到干净的codes

```python
def generate(self, batch, n_return_sequences=1):
    batch_size = batch['input_ids'].shape[0]
    
    # 1. 初始化：所有codes都是MASK
    current_codes = torch.full(
        (batch_size, 32), 
        self.mask_token_id, 
        device=self.device
    )
    
    # 2. 从T到1逐步去噪
    for t in reversed(range(1, self.T + 1)):
        # 构建输入
        input_ids = torch.cat([batch['input_ids'], current_codes.unsqueeze(1)], dim=1)
        time_emb = self.time_embed(torch.full((batch_size,), t, device=self.device))
        
        # 前向传播
        input_tokens = self.item_id2tokens[input_ids]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        input_embs[:, -1, :] += time_emb
        
        outputs = self.gpt2(inputs_embeds=input_embs)
        final_states = self.pred_heads(outputs.last_hidden_state[:, -1, :])
        
        # 预测每个位置的code
        predicted_codes = []
        for i in range(32):
            logits = torch.matmul(
                F.normalize(final_states[:, i, :], dim=-1),
                F.normalize(self.gpt2.wte.weight[1:-1], dim=-1).T
            )
            predicted_codes.append(logits.argmax(dim=-1))
        
        predicted_codes = torch.stack(predicted_codes, dim=1)
        
        # 3. 更新：保留已预测的，继续mask未到时间的
        mask_ratio = (t - 1) / self.T
        num_to_keep = int(32 * (1 - mask_ratio))
        
        # 选择置信度最高的codes保留
        confidence = torch.stack([
            F.softmax(torch.matmul(
                F.normalize(final_states[:, i, :], dim=-1),
                F.normalize(self.gpt2.wte.weight[1:-1], dim=-1).T
            ), dim=-1).max(dim=-1)[0]
            for i in range(32)
        ], dim=1)
        
        top_k_indices = confidence.topk(num_to_keep, dim=1).indices
        
        # 更新current_codes
        new_codes = current_codes.clone()
        for b in range(batch_size):
            for idx in top_k_indices[b]:
                new_codes[b, idx] = predicted_codes[b, idx]
        
        current_codes = new_codes
    
    # 4. 将codes转换为item IDs
    item_logits = self.codes_to_item_logits(current_codes)
    preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
    
    return preds.unsqueeze(-1)
```

## 5. 关键改进点

### 5.1 与传统Autoregressive的区别

| 维度 | Autoregressive (RPG) | Diffusion (LLaDA-style) |
|------|---------------------|------------------------|
| 预测方式 | 顺序预测32个codes | 并行预测所有codes |
| 训练目标 | 最大化p(next\|history) | 最大化p(clean\|noisy) |
| 推理速度 | 1次forward | T次forward (但可并行) |
| 多样性 | 低（贪心解码） | 高（随机采样） |
| 可控性 | 难 | 易（条件生成） |

### 5.2 LLaDA特有优势

1. **并行预测**: 32个codes同时预测，不存在error accumulation
2. **灵活采样**: 可以控制不同位置的噪声水平
3. **部分生成**: 可以fix某些codes，只生成其他codes
4. **双向依赖**: 不像autoregressive只能从左到右

### 5.3 推荐系统特定优化

```python
# 条件生成：固定某些属性
def conditional_generate(self, batch, fixed_codes=None, fixed_positions=None):
    """
    fixed_codes: (batch, k) - 固定的codes
    fixed_positions: (batch, k) - 哪些位置固定
    
    例如：固定前16个codes（控制大类），生成后16个codes（细节）
    """
    current_codes = torch.full((batch_size, 32), self.mask_token_id)
    
    # 设置固定的codes
    if fixed_codes is not None:
        for b in range(batch_size):
            current_codes[b, fixed_positions[b]] = fixed_codes[b]
    
    # 只去噪未固定的位置
    for t in reversed(range(1, self.T + 1)):
        # ... 推理过程 ...
        # 但保持fixed_positions的值不变
        pass
```

## 6. 实现优先级

### Phase 1: 基础实现
- [ ] 添加MASK token到vocabulary
- [ ] 实现forward_diffusion函数
- [ ] 修改training loop支持随机时间步采样
- [ ] 实现基础的reverse process

### Phase 2: 优化
- [ ] 实现不同的masking schedule (cosine, square, etc.)
- [ ] 添加时间步embedding的不同注入方式
- [ ] 优化inference速度（并行、缓存等）
- [ ] 实现confidence-based sampling

### Phase 3: 高级功能
- [ ] 条件生成（固定某些codes）
- [ ] 多样性采样（temperature, top-p）
- [ ] 与图搜索结合
- [ ] A/B测试对比autoregressive

## 7. 预期效果

### 优势
- ✅ **多样性提升**: 可以生成更diverse的推荐
- ✅ **可控性**: 可以控制生成过程（如固定类别）
- ✅ **并行预测**: 避免error propagation
- ✅ **理论保证**: 有likelihood bound的理论支撑

### 挑战
- ⚠️ **推理速度**: 需要T次forward (但T可以较小，如10-20)
- ⚠️ **训练稳定性**: Diffusion训练可能需要更多tricks
- ⚠️ **超参数**: 需要调整masking schedule, T等

## 8. 实验计划

### 8.1 消融实验
1. **T的影响**: T=10, 20, 32, 50
2. **Masking schedule**: linear vs cosine vs square
3. **时间步embedding**: 加法 vs 连接 vs cross-attention
4. **采样策略**: greedy vs confidence-based vs random

### 8.2 对比实验
- RPG (autoregressive) vs LLaDA-RPG (diffusion)
- 指标: NDCG, Recall, Diversity, Coverage

### 8.3 Case Study
- 可控生成：固定类别，生成具体item
- 多样性：同一用户生成多个不同推荐
- 冷启动：对新item的泛化能力

## 9. 参考资源

- 论文: [LLaDA: Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- 项目主页: [待补充]
- 相关工作:
  - D3PM (Discrete Denoising Diffusion Probabilistic Models)
  - Diffusion-LM
  - DiffuSeq

## 10. 下一步

1. 实现`LLaDARecommender`类
2. 修改`tokenizer.py`添加MASK token
3. 修改`trainer.py`支持diffusion训练
4. 在Beauty数据集上进行pilot实验

