# LLADA 模型架构与训练详解

## 概述

LLADA (Large Language Diffusion for Recommendation) 是一个基于扩散模型的推荐系统，受到 [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) 启发。

**核心思想**：将next-token自回归预测改为mask-predict扩散过程
- RPG：自回归预测下一个item的所有tokens
- LLADA：从全mask状态逐步去噪，预测被mask的tokens

---

## 一、训练样本选择（继承自RPG）

### 样本生成策略

LLADA **完全继承** RPG的tokenizer和样本构造策略（通过`class LLADATokenizer(RPGTokenizer)`）

#### 1. 短序列（≤ 51个items）
```python
# 用户序列: [A, B, C, D, E]
输入: [A, B, C, D]
标签: [B, C, D, E]  # 每个位置预测下一个item
```

#### 2. 长序列（> 51个items）
```python
# 用户序列有100个items

# 样本1: 前51个，全序列预测
输入: [item_1, ..., item_50]
标签: [item_2, ..., item_51]

# 样本2-50: 滑动窗口，只预测最后一个
输入: [item_2, ..., item_51]
标签: 只预测 item_52 (最后一个位置)
...
```

### Tokenizer特性
```python
# 继承RPG的所有功能
- 复用预训练的OPQ semantic IDs（不需要重新训练！）
- 相同的item到tokens映射
- 唯一差别：添加一个 [MASK] token

# Vocabulary结构:
0:          [PAD]
1-256:      digit 0 codes (codebook 0)
257-512:    digit 1 codes (codebook 1)
...
7937-8192:  digit 31 codes (codebook 31)
8193:       [EOS]
8194:       [MASK]  ← LLADA新增
```

---

## 二、模型结构详解

### 1. 整体架构

```
训练时 (Diffusion Training):
用户历史序列 [item_1, ..., item_k]
        ↓
目标item的tokens (32个) → 采样timestep t → 按mask ratio mask部分tokens
        ↓
历史items的embeddings + 被mask的目标item embeddings
        ↓
加入时间嵌入 (time step t)
        ↓
GPT2 Transformer编码
        ↓
32个预测头 (每个codebook一个ResBlock)
        ↓
预测被mask的tokens
        ↓
Loss: 只在被mask的位置计算CrossEntropy


推理时 (Iterative Denoising):
初始状态: 所有32个tokens都是 [MASK]
        ↓
For t = T, T-1, ..., 1:
    用户历史 + 当前状态的tokens → GPT2 + 时间嵌入
        ↓
    预测所有32个tokens
        ↓
    保留top-k个最confident的预测
    其余继续保持 [MASK]
        ↓
最终: 得到32个干净的tokens → 映射到item
```

### 2. 与GPT2的关系

#### LLADA在GPT2基础上的修改：

**继承自GPT2：**
- ✅ `GPT2Model`: Transformer编码器
- ✅ `gpt2.wte`: Token Embedding层
- ✅ 自注意力机制
- ✅ Position embeddings

**LLADA的创新：**
- ➕ **时间嵌入层**: `time_embed` (T+1个时间步的embedding)
- ➕ **Mask token**: vocab中新增一个特殊token
- ➕ **扩散训练**: 随机mask部分tokens进行训练
- ➕ **迭代去噪**: 推理时从全mask状态逐步恢复
- ✅ 保留32个预测头（和RPG一样）

#### 配置对比：

```python
# LLADA的GPT2配置（来自config.yaml）
n_embd = 448      # embedding维度
n_layer = 2       # 2层Transformer
n_head = 4        # 4个注意力头
n_inner = 1024    # FFN中间层
T = 32            # 扩散步数

# 对比：
GPT2-small:  12层, 768维, 117M参数
LLADA:       2层, 448维, ~5M参数 + 时间embedding
```

---

## 三、训练过程详解

### Forward Pass（训练时）

```python
def forward(self, batch):
    # 输入数据
    batch = {
        'input_ids': [101, 205, 307],      # 历史item IDs
        'attention_mask': [1, 1, 1],
        'labels': [410],                    # 目标item ID
        'seq_lens': 3
    }
    
    # Step 1: 采样时间步
    t = torch.randint(1, T+1, (batch_size,))  # 例如: t=15
    
    # Step 2: 获取目标item的tokens
    target_codes = item_id2tokens[410]  
    # → [15, 278, 530, ..., 8100]  (32个tokens)
    
    # Step 3: Forward Diffusion（前向加噪）
    # 根据t计算mask ratio
    mask_ratio = t / T  # 例如: 15/32 = 0.47
    num_masked = int(32 * 0.47)  # ≈ 15个
    
    # 随机选择15个位置mask掉
    masked_codes = target_codes.clone()
    masked_positions = [3, 7, 10, 12, ...]  # 随机选15个
    masked_codes[masked_positions] = MASK_TOKEN  # 8194
    # → [15, 278, 530, MASK, ..., MASK, 8100]
    
    # Step 4: 编码历史序列
    input_tokens = item_id2tokens[batch['input_ids']]
    # → shape: (batch=1, seq_len=3, n_digit=32)
    input_embs = gpt2.wte(input_tokens).mean(dim=-2)
    # → shape: (batch=1, seq_len=3, n_embd=448)
    
    # Step 5: GPT2编码
    outputs = gpt2(inputs_embeds=input_embs, attention_mask=[1,1,1])
    # → last_hidden_state: (1, 3, 448)
    
    # Step 6: 32个预测头处理
    final_states = []
    for i in range(32):
        state = pred_heads[i](outputs.last_hidden_state)
        final_states.append(state.unsqueeze(-2))
    final_states = torch.cat(final_states, dim=-2)
    # → shape: (1, 3, 32, 448)
    
    # Step 7: 提取最后一个位置的状态（预测目标item）
    selected_states = final_states[:, -1, :, :]  # (1, 32, 448)
    
    # Step 8: 加入时间嵌入
    time_emb = time_embed(t)  # (1, 448)
    selected_states = selected_states + time_emb.unsqueeze(1)
    # → shape: (1, 32, 448)
    
    # Step 9: 归一化并计算logits
    selected_states_norm = F.normalize(selected_states, dim=-1)
    token_emb_norm = F.normalize(gpt2.wte.weight[1:8193], dim=-1)
    token_embs = torch.chunk(token_emb_norm, 32, dim=0)
    # → 32个tensor，每个 (256, 448)
    
    # Step 10: 计算每个codebook的loss
    losses = []
    for i in range(32):
        logits = torch.matmul(selected_states_norm[:, i, :], 
                             token_embs[i].T) / temperature
        # → shape: (1, 256)
        
        # 获取ground truth label
        label = target_codes[i] - i * 256 - 1  # 转为局部ID (0-255)
        
        # 计算交叉熵
        loss_i = CrossEntropyLoss(logits, label)
        losses.append(loss_i)
    
    # Step 11: 平均所有codebook的loss
    total_loss = mean(losses)  # 32个loss的平均
    
    return total_loss
```

### 关键点：

1. **时间步采样**：每个训练样本随机采样一个t ∈ [1, T]
2. **Mask Schedule**：
   - Linear: `mask_ratio = t / T`
   - Cosine: `mask_ratio = cos(π * t / 2T)`
   - Square: `mask_ratio = (t / T)²`
3. **只在有label的位置计算loss**（和RPG一样）
4. **时间嵌入加到states上**（区别于RPG）

---

## 四、推理过程详解

### Iterative Denoising（迭代去噪）

```python
def generate(self, batch, n_return_sequences=20):
    # 初始化：所有tokens都是MASK
    current_codes = torch.full((batch_size, 32), MASK_TOKEN)
    # → [MASK, MASK, MASK, ..., MASK]  (32个MASK)
    
    # 从t=T逐步去噪到t=1
    for t in reversed(range(1, T+1)):  # T=32, ..., 2, 1
        # Step 1: 编码历史序列
        input_tokens = item_id2tokens[batch['input_ids']]
        input_embs = gpt2.wte(input_tokens).mean(dim=-2)
        # → (batch, seq_len, 448)
        
        # Step 2: 当前codes的embedding
        target_embs = gpt2.wte(current_codes).mean(dim=1, keepdim=True)
        # → (batch, 1, 448)
        
        # Step 3: 拼接历史 + 当前状态
        all_embs = torch.cat([input_embs, target_embs], dim=1)
        # → (batch, seq_len+1, 448)
        
        # Step 4: 加时间嵌入（只在最后一个位置）
        time_emb = time_embed(t)
        all_embs[:, -1, :] = all_embs[:, -1, :] + time_emb
        
        # Step 5: GPT2前向
        outputs = gpt2(inputs_embeds=all_embs)
        target_hidden = outputs.last_hidden_state[:, -1, :]
        # → (batch, 448)
        
        # Step 6: 32个预测头预测
        final_states = []
        for i in range(32):
            state = pred_heads[i](target_hidden)
            final_states.append(state.unsqueeze(1))
        final_states = torch.cat(final_states, dim=1)
        # → (batch, 32, 448)
        
        # Step 7: 预测所有32个tokens
        predicted_codes = []
        confidence_scores = []
        
        for i in range(32):
            logits = torch.matmul(
                F.normalize(final_states[:, i, :], dim=-1),
                F.normalize(token_embs[i], dim=-1).T
            ) / temperature
            
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)
            
            predicted_code = pred_idx + i * 256 + 1
            predicted_codes.append(predicted_code)
            confidence_scores.append(confidence)
        
        predicted_codes = torch.stack(predicted_codes, dim=1)
        # → (batch, 32)
        confidence_scores = torch.stack(confidence_scores, dim=1)
        # → (batch, 32)
        
        # Step 8: 更新策略（保留高置信度的预测）
        if t > 1:
            # 根据schedule计算应该保留多少个codes
            mask_ratio = (t-1) / T
            num_to_keep = int(32 * (1 - mask_ratio))
            
            # 选择top-k个最confident的预测
            _, top_k_indices = confidence_scores.topk(num_to_keep, dim=1)
            
            # 只更新那些还是MASK的位置
            for b in range(batch_size):
                for idx in top_k_indices[b]:
                    if current_codes[b, idx] == MASK_TOKEN:
                        current_codes[b, idx] = predicted_codes[b, idx]
        else:
            # 最后一步：使用所有预测
            current_codes = predicted_codes
    
    # Step 9: 将codes映射到items
    # 方法1: Count matching (默认)
    item_logits = []
    for b in range(batch_size):
        matches = (current_codes[b].unsqueeze(0) == all_item_codes).sum(dim=1)
        item_logits.append(matches.float())
    item_logits = torch.stack(item_logits)
    
    # 方法2: Embedding similarity
    # code_embs = gpt2.wte(current_codes).mean(dim=1)
    # all_item_embs = gpt2.wte(all_item_codes).mean(dim=1)
    # item_logits = cosine_similarity(code_embs, all_item_embs)
    
    # Step 10: TopK选择
    preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
    
    return preds
```

### 去噪示例（T=32）：

```python
# 初始状态 (t=32):
[MASK, MASK, MASK, ..., MASK]  # 32个MASK

# t=32预测后，保留top-1个最confident:
[15, MASK, MASK, ..., MASK]    # 只确定第1个code

# t=31预测后，保留top-2个:
[15, 278, MASK, ..., MASK]     # 确定前2个

# t=30:
[15, 278, 530, MASK, ..., MASK]

# ...逐步去噪...

# t=1 (最后一步):
[15, 278, 530, 789, ..., 8100]  # 所有32个codes都确定

# 映射到item:
这32个codes和哪个item的codes匹配最多？→ 推荐该item
```

---

## 五、与RPG的对比

| 维度 | RPG | LLADA |
|-----|-----|-------|
| **训练样本** | 滑动窗口 + 多位置预测 | 相同（继承） |
| **Semantic IDs** | OPQ预训练 | 相同（继承） |
| **模型骨架** | GPT2 (2层, 448维) | GPG2 (2层, 448维) |
| **预测头** | 32个ResBlock | 32个ResBlock |
| **训练范式** | Next-token预测 | **Mask-predict扩散** |
| **时间嵌入** | ❌ 无 | ✅ 有 (T+1维) |
| **MASK token** | ❌ 无 | ✅ 有 (vocab+1) |
| **训练时采样** | 直接预测 | **随机mask + 时间步采样** |
| **推理方式** | 并行预测32个tokens | **迭代去噪T步** |
| **推理速度** | 1次forward | T次forward (慢32倍) |
| **生成质量** | 固定 | 可能更robust（多次refine） |

---

## 六、关键设计细节

### 1. Mask Schedule的影响

```python
# Linear (默认)
t=32: mask 100%的codes (全MASK)
t=16: mask 50%的codes
t=1:  mask 3%的codes

# Cosine (更平滑)
t=32: mask 100%
t=16: mask 70%  # 前期mask更多
t=1:  mask 1%

# Square (激进)
t=32: mask 100%
t=16: mask 25%  # 前期快速确定
t=1:  mask 0%
```

### 2. Codes到Items的映射

```python
# 方法1: Count Matching (默认)
item_score = sum(generated_codes == item_codes)
# 优点: 简单直接
# 缺点: 一个code错误就无法exact match

# 方法2: Embedding Similarity
item_score = cosine_similarity(
    mean(wte[generated_codes]),
    mean(wte[item_codes])
)
# 优点: 对错误更robust
# 缺点: 可能匹配到相似但不正确的item

# 方法3: Hybrid (可选)
item_score = α * count + β * similarity
```

### 3. 训练时的注意事项

- **标签处理**：只在有效label位置计算loss（过滤-100和0）
- **时间步均匀采样**：确保所有t∈[1,T]都被训练到
- **Mask随机性**：每次训练同一个样本可能mask不同位置

---

## 七、示例：完整流程

### 训练样本：
```python
用户序列: [A, B, C, D]
目标item: E (id=410)
item E的codes: [15, 278, 530, 789, ..., 8100]

# 采样 t=20, mask_ratio=20/32=0.625
mask 20个位置: [15, MASK, 530, MASK, ..., MASK]

# 输入GPT2:
历史: [A, B, C, D] → embeddings
时间: t=20 → time_embed(20)

# 目标:
预测那20个被mask的位置的正确token
loss = mean(CrossEntropy[被mask的20个位置])
```

### 推理：
```python
用户历史: [A, B, C, D]

# 初始: [MASK×32]
# t=32: 预测→保留1个
# t=31: 预测→保留2个
# ...
# t=1:  预测→得到32个完整codes

# 映射: [15, 278, 530, ...] → item E
# 推荐: E
```

---

## 八、总结

**LLADA = GPT2骨架 + Diffusion训练 + 迭代去噪**

1. **继承RPG**：样本构造、semantic IDs、模型结构基本相同
2. **核心创新**：从自回归改为扩散，mask-predict范式
3. **推理权衡**：更多计算（T次forward）换取可能更好的生成质量
4. **灵活性**：可以调整T、mask schedule、codes-to-item映射策略

