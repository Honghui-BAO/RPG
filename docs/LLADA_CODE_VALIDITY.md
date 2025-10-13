# LLADA代码合法率统计

## 问题背景

LLADA通过diffusion生成32个semantic codes，但这些codes的组合可能在实际的item库中**不存在**。

例如：
```python
生成的codes: [16, 203, 78, 145, ..., 199]
实际item库中可能没有完全匹配这个组合的item
```

因此我们需要统计**代码合法率（Code Validity）**。

## 统计指标

### 1. **Exact Match Rate（完全匹配率）**

生成的32个codes**完全匹配**某个真实item的比例。

```python
例如：100个生成的codes中
- 23个完全匹配真实item
- 77个不存在于item库

Exact Match Rate = 23/100 = 0.23 (23%)
```

### 2. **Average Max Matching Codes（平均最大匹配数）**

对于不完全匹配的codes，与最相似item的匹配code数量。

```python
生成: [16, 203, 78, 145, ..., 199]
最相似item: [16, 203, 78, 150, ..., 199]  # 31个匹配
              ✓   ✓    ✓   ✗         ✓

Avg Max Match = 平均每个生成codes与最相似item的匹配数
理想值: 32 (完全匹配)
```

## 使用方法

### 自动统计（训练时）

```bash
./scripts/train_llada.sh
```

评估时会自动输出：
```
Val Results: {
    'ndcg@10': 0.1234,
    'recall@10': 0.2345,
    'code_exact_match_rate': 0.23,      # ← 新增！
    'code_avg_max_match': 28.5,         # ← 新增！
}
```

### 手动检查

```python
from genrec.models.LLADA.model import LLaDARecommender

# 生成推荐
preds, codes, validity = model.generate(batch, return_codes=True)

print(validity)
# {
#     'exact_match_rate': 0.23,
#     'avg_max_matching_codes': 28.5,
#     'exact_matches': 23,
#     'total': 100
# }
```

### 分析单个生成结果

```python
# 检查某个生成的codes
codes = tensor([[16, 203, 78, 145, ..., 199]])  # (1, 32)

validity = model.check_code_validity(codes)
print(f"Exact match: {validity['exact_match_rate']}")
print(f"Max matching codes: {validity['avg_max_matching_codes']}")

# 如果不是完全匹配，找到最相似的item
all_item_codes = model.item_id2tokens[1:]
matches = (codes[0].unsqueeze(0) == all_item_codes).sum(dim=1)
best_match_idx = matches.argmax()
best_match_codes = all_item_codes[best_match_idx]

print(f"Generated:     {codes[0]}")
print(f"Best match:    {best_match_codes}")
print(f"Matching codes: {matches.max()}/32")
```

## 预期结果

### 初期训练（Epoch 1-10）
```
code_exact_match_rate: 0.05 - 0.15  # 5-15%完全匹配
code_avg_max_match: 20 - 25         # 平均匹配20-25个codes
```

### 充分训练后（Epoch 50+）
```
code_exact_match_rate: 0.30 - 0.50  # 30-50%完全匹配
code_avg_max_match: 28 - 31         # 平均匹配28-31个codes
```

### 理想情况
```
code_exact_match_rate: 1.0          # 100%完全匹配
code_avg_max_match: 32.0            # 所有codes都匹配
```

## 与RPG的对比

| 模型 | Exact Match Rate | 说明 |
|------|------------------|------|
| **RPG** | ~100% | 直接从item库选择，总是合法 |
| **LLADA** | 20-50%? | 生成新的codes组合，可能不存在 |

## 意义

### 如果合法率很高（>80%）
✅ **说明diffusion学到了有效的codes空间**
- 可以直接使用生成的codes
- Diffusion是可行的

### 如果合法率较低（<30%）
⚠️ **说明存在问题**
- codes空间过大，组合太多
- 需要改进：
  1. 使用更强的regularization
  2. 减少n_codebook（如从32降到16）
  3. 使用连续diffusion而不是离散diffusion
  4. 改进codes到item的映射方式

## 优化建议

### 方法1：改进映射函数
```python
def _codes_to_item_logits(self, codes):
    # 当前：数codes匹配个数
    # 改进：使用embedding相似度（更平滑）
    
    code_embs = self.gpt2.wte(codes).mean(dim=1)  # (batch, 768)
    all_item_embs = ...  # (n_items, 768)
    logits = cosine_similarity(code_embs, all_item_embs)
    return logits
```

### 方法2：减少codes数量
```yaml
# 从32降到16
n_codebook: 16  # 而不是32
# 这样codes空间从256^32降到256^16，更容易match
```

### 方法3：Constrained Decoding
```python
# 在生成时，只选择真实存在的codes组合
# 类似beam search with constraint
```

## 实验建议

1. **跟踪训练过程中的合法率变化**
   - Epoch 1, 5, 10, 20, 50, 100
   - 看是否随训练提高

2. **对比不同配置**
   - T=10 vs 20 vs 32
   - n_codebook=16 vs 32
   - mask_schedule: linear vs cosine

3. **分析不匹配的cases**
   - 哪些codes位置最容易错？
   - 不匹配的codes与最相似item差多少？

## 可视化

建议创建可视化脚本：
```python
import matplotlib.pyplot as plt

# 训练过程中的合法率
epochs = [1, 5, 10, 20, 50]
exact_match_rates = [0.05, 0.12, 0.18, 0.25, 0.35]

plt.plot(epochs, exact_match_rates)
plt.xlabel('Epoch')
plt.ylabel('Exact Match Rate')
plt.title('LLADA Code Validity over Training')
```

