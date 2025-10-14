# LLADA with BERT Backbone

## 为什么用BERT替代GPT2?

### GPT2的局限
```python
# GPT2使用Causal (单向) Attention
position 1: 只能看 [item_1]
position 2: 只能看 [item_1, item_2]
position 3: 只能看 [item_1, item_2, item_3]

# 对于推荐任务的问题：
- 预测item_4时，理论上应该综合考虑整个序列
- 但GPT2的position 3只能看左边，看不到右边的信息
- Causal mask是为文本生成设计的，推荐不需要
```

### BERT的优势
```python
# BERT使用Bidirectional (双向) Attention  
所有positions都能看到整个序列！

position 1: 可以看 [item_1, item_2, item_3]
position 2: 可以看 [item_1, item_2, item_3]
position 3: 可以看 [item_1, item_2, item_3]

# 对推荐任务的好处：
✅ 更全面的序列理解
✅ 更好的用户行为建模
✅ 每个位置都能获得全局信息
```

## 关键改动

### 代码层面
```python
# Before (GPT2)
from transformers import GPT2Config, GPT2Model
self.gpt2 = GPT2Model(gpt2config)
embeddings = self.gpt2.wte(tokens)

# After (BERT)
from transformers import BertConfig, BertModel
self.encoder = BertModel(bert_config)
embeddings = self.encoder.embeddings.word_embeddings(tokens)
```

### Attention机制
```python
# GPT2: Causal Mask
Attention Matrix (下三角):
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]

# BERT: Full Attention
Attention Matrix (全1):
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
```

## 预期效果

### 优势
1. **更好的序列建模**
   - 双向attention捕捉全局模式
   - 适合推荐任务的特性

2. **理论更合理**
   - 推荐不是文本生成，不需要causal约束
   - 可以同时考虑前后item的关系

3. **可能更好的性能**
   - 更丰富的表示
   - 更好的泛化能力

### 可能的问题
1. **训练稳定性**
   - BERT可能需要不同的学习率
   - 可能需要调整warmup steps

2. **计算量**
   - Full attention vs causal attention
   - 理论上略慢，但差异不大

## 实验设置

```yaml
# Branch: llada_backbone
backbone: BERT
n_codebook: 32  # 先用32测试
code_to_item_method: count
```

## 对比实验

| 配置 | Backbone | Attention | 预期NDCG@10 |
|------|----------|-----------|------------|
| llada | GPT2 | Causal | baseline |
| llada_backbone | BERT | Bidirectional | +2-5%? |

## 使用方法

```bash
git checkout llada_backbone
./scripts/train_llada.sh
```

## 后续探索

如果BERT效果好：
- 可以结合到len8/len4版本
- 测试其他backbone (RoBERTa, ALBERT, etc.)
- 对比不同层数的BERT

