# RPG 模型架构与训练样本详解

## 一、训练样本选择策略

### 1. 数据格式
每个用户有一个交互序列：`[item_1, item_2, item_3, ..., item_n]`

### 2. 训练集样本生成（滑动窗口 + 全序列预测）

RPG使用了两种策略来充分利用用户序列：

#### 策略A：短序列 - 全序列预测
**条件**：`len(item_seq) <= max_item_seq_len + 1`（例如：序列长度 ≤ 51）

```python
# 例如：用户序列 [A, B, C, D, E]，max_item_seq_len=50
输入：[A, B, C, D]
标签：[B, C, D, E]  # 每个位置都预测下一个item
```

**特点**：
- 一次forward计算多个损失（每个位置预测下一个item）
- 类似语言模型的自回归训练
- 代码：`_tokenize_first_n_items()`

#### 策略B：长序列 - 滑动窗口
**条件**：`len(item_seq) > max_item_seq_len + 1`（例如：序列长度 > 51）

```python
# 例如：用户序列有100个items，max_item_seq_len=50

# 样本1：前51个
输入：[item_1, ..., item_50]
标签：[item_2, ..., item_51]  # 全序列预测

# 样本2：[item_2, ..., item_52]
输入：[item_2, ..., item_51]
标签：只预测 item_52  # 只预测最后一个

# 样本3：[item_3, ..., item_53]
输入：[item_3, ..., item_52]
标签：只预测 item_53

# ...
# 样本50：[item_50, ..., item_100]
输入：[item_50, ..., item_99]
标签：只预测 item_100
```

**特点**：
- 滑动窗口生成多个训练样本
- 除了第一个样本，其他只预测最后一个item（避免重复）
- 充分利用长序列信息
- 代码：`_tokenize_later_items()`

### 3. 验证/测试集样本生成

```python
# 只取最后 max_item_seq_len+1 个items
输入：[item_{n-50}, ..., item_{n-1}]
标签：只预测 item_n  # 只评估最后一个item的预测
```

---

## 二、模型结构详解

### 1. 整体架构

```
用户序列 [item_1, item_2, ..., item_k]
    ↓
将每个item映射为多个tokens（通过预训练的OPQ）
    ↓
Token Embeddings (从GPT2的wte)
    ↓
对每个item的tokens取平均 → 得到item embedding
    ↓
GPT2 Transformer (2层，448维)
    ↓
多个预测头（每个codebook一个ResBlock）
    ↓
计算与token embeddings的相似度
    ↓
预测每个codebook的token
```

### 2. 与GPT2的关系

#### GPT2在RPG中的作用：

**使用的部分**：
- ✅ `GPT2Model`：只用Transformer编码器部分（不含LM head）
- ✅ `gpt2.wte`：Word Token Embedding层（实际上是Item Token Embedding）
- ✅ Transformer layers：自注意力机制建模序列

**不使用的部分**：
- ❌ GPT2的语言模型头（用自定义的预测头代替）
- ❌ GPT2的自回归解码（推理时并行预测）

#### 具体配置（来自config.yaml）：

```python
# RPG的GPT2配置
vocab_size = n_codebook * codebook_size + 2  # 例如：32*256+2 = 8194
n_positions = 50  # 最大序列长度
n_embd = 448      # embedding维度（GPT2-small是768）
n_layer = 2       # 只有2层！（GPT2-small是12层）
n_head = 4        # 4个注意力头（GPT2-small是12个）
n_inner = 1024    # FFN中间层
```

**对比**：
- GPT2-small：12层，768维，117M参数
- RPG：2层，448维，约5M参数（不含embedding）

### 3. 详细的Forward流程

```python
def forward(self, batch):
    # Step 1: 获取每个item的tokens
    # batch['input_ids']: (batch_size, seq_len) - item IDs
    # item_id2tokens: (n_items, n_codebook) - 预训练的token映射
    input_tokens = self.item_id2tokens[batch['input_ids']]
    # → (batch_size, seq_len, n_codebook)  例如：(32, 50, 32)
    
    # Step 2: Token embeddings并取平均
    # gpt2.wte.weight: (vocab_size=8194, n_embd=448)
    input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
    # → (batch_size, seq_len, n_codebook, n_embd) 
    # → mean → (batch_size, seq_len, n_embd)
    
    # Step 3: GPT2编码
    outputs = self.gpt2(
        inputs_embeds=input_embs,
        attention_mask=batch['attention_mask']
    )
    # → outputs.last_hidden_state: (batch_size, seq_len, n_embd)
    
    # Step 4: 多个预测头
    # 每个codebook有一个独立的ResBlock
    final_states = []
    for i in range(self.n_pred_head):  # n_pred_head = n_codebook = 32
        state = self.pred_heads[i](outputs.last_hidden_state)
        final_states.append(state.unsqueeze(-2))
    final_states = torch.cat(final_states, dim=-2)
    # → (batch_size, seq_len, n_pred_head=32, n_embd=448)
    
    # Step 5: 计算损失（训练时）
    if return_loss:
        # 5.1 提取需要预测的位置
        label_mask = batch['labels'].view(-1) != -100
        selected_states = final_states.view(-1, 32, 448)[label_mask]
        # → (n_valid_labels, 32, 448)
        
        # 5.2 归一化
        selected_states = F.normalize(selected_states, dim=-1)
        selected_states = torch.chunk(selected_states, 32, dim=1)
        
        # 5.3 Token embeddings归一化
        token_emb = self.gpt2.wte.weight[1:-1]  # 去除BOS/EOS
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, 32, dim=0)
        # → 32个tensor，每个 (256, 448)
        
        # 5.4 计算每个预测头的logits
        token_logits = []
        for i in range(32):
            logit = torch.matmul(selected_states[i].squeeze(1), 
                                token_embs[i].T) / self.temperature
            # → (n_valid_labels, 256)
            token_logits.append(logit)
        
        # 5.5 计算每个头的交叉熵损失
        token_labels = self.item_id2tokens[batch['labels']]
        # → (n_valid_labels, 32)
        
        losses = []
        for i in range(32):
            # 将全局token ID转换为局部ID（0-255）
            local_label = token_labels[:, i] - i * 256 - 1
            loss = CrossEntropyLoss(token_logits[i], local_label)
            losses.append(loss)
        
        # 5.6 平均所有头的损失
        total_loss = torch.mean(torch.stack(losses))
    
    return outputs
```

---

## 三、关键设计特点

### 1. Item表示方式
- 每个item用32个离散tokens表示（通过OPQ预训练）
- 每个token来自256个候选（8位编码）
- 总共：32 × 256 = 8192种token组合

### 2. 序列编码方式
- **不是**直接编码token序列（会很长：seq_len × 32）
- **而是**先将32个tokens的embedding平均，得到item embedding
- 然后用GPT2编码item序列

### 3. 预测方式
- 32个预测头并行预测32个tokens
- 每个头独立优化（解耦）
- 推理时可以并行计算（不需要自回归）

### 4. 与传统GPT2的区别

| 维度 | GPT2（语言模型） | RPG |
|-----|-----------------|-----|
| 输入 | Token序列 | Item序列（每个item由多个tokens平均表示） |
| 输出 | 预测下一个token | 预测下一个item的所有tokens（并行） |
| 解码 | 自回归（sequential） | 并行（parallel） |
| Vocab | 50K+ words | 8K+ item tokens |
| 层数 | 12层 | 2层 |
| 用途 | 文本生成 | 序列推荐 |

---

## 四、示例：完整训练样本

假设：
- 用户序列：`[item_A, item_B, item_C, item_D, item_E]`
- item_A的tokens：`[15, 278, 530, 789, ..., 8100]`（32个）
- max_item_seq_len = 50

### 训练样本：
```
输入序列（item IDs）：[A, B, C, D]
注意力掩码：[1, 1, 1, 1]
标签（item IDs）：[B, C, D, E]

转换为tokens后：
- A → [15, 278, 530, ..., 8100]
- 平均后得到 embedding_A

GPT2输入：[embedding_A, embedding_B, embedding_C, embedding_D]
预测目标：item_B, item_C, item_D, item_E 的32个tokens
```

### 损失计算：
- 位置0预测item_B：32个token的交叉熵平均
- 位置1预测item_C：32个token的交叉熵平均
- 位置2预测item_D：32个token的交叉熵平均
- 位置3预测item_E：32个token的交叉熵平均
- 总损失：4个位置损失的平均

---

## 五、总结

**RPG = 轻量级GPT2骨架 + 多头token预测 + item表示压缩**

1. 借用GPT2的Transformer编码能力
2. 但大幅简化（2层 vs 12层）
3. 输入是压缩的item表示（32个tokens平均）
4. 输出是并行的多token预测（不需要sequential decoding）
5. 训练时充分利用序列（滑动窗口 + 多位置预测）

