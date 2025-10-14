# LLADA Length-8 Experiment

## 实验目的

将semantic codes长度从32降到8，探索：
1. **训练/推理速度**是否显著提升
2. **Code validity**是否提高（组合空间更小）
3. **推荐质量**是否有显著变化

## 关键改变

| 参数 | 原版(32) | 新版(8) | 影响 |
|------|---------|---------|------|
| `n_codebook` | 32 | 8 | Codes长度 |
| `diffusion_steps` | 32 | 8 | 去噪步数 |
| `codes_per_step` | 8 | 4 | 每步确定的codes数 |
| **Vocab size** | 8195 | 2051 | (8×256+1+1+1) |
| **Codes组合空间** | 256^32 | 256^8 | 减少约10^58倍！ |
| **推理步数** | 4步 | 2步 | 再快一倍 |

## 理论分析

### 优势

**1. 速度大幅提升**
```python
# 32 codes版本
- Vocab: 8195个tokens
- Embedding: 8195 × 768 = 6M+ params
- 推理: 4步 × 32个预测头

# 8 codes版本  
- Vocab: 2051个tokens
- Embedding: 2051 × 768 = 1.5M params (减少75%)
- 推理: 2步 × 8个预测头 (减少75%)
```

**2. Code Validity提升**
```python
# 组合空间
32 codes: 256^32 ≈ 1.3 × 10^77 种组合
8 codes:  256^8  ≈ 1.8 × 10^19 种组合

# 实际items: ~12000个
32 codes: 几乎不可能随机生成真实item
8 codes:  更有可能生成真实item (但仍然很难)
```

**3. 训练更容易**
- 更小的搜索空间
- 更少的参数
- 可能收敛更快

### 劣势

**1. 表达能力下降**
```python
# 信息压缩
32 codes: 每个item用32×8=256 bits表示
8 codes:  每个item用8×8=64 bits表示
→ 信息量减少75%
```

**2. Item区分度降低**
```python
# 12000个items，8个codes
平均每个code组合对应: 12000 / 256^8 ≈ 0个
但由于codes不是均匀分布，可能出现碰撞
```

## 实验设置

### 训练
```bash
./scripts/train_llada.sh
# n_codebook=8
# diffusion_steps=8
# codes_per_step=4
```

### 对比基准
- **RPG-32**: 原版RPG (32 codes)
- **LLADA-32**: LLADA with 32 codes (llada分支)
- **LLADA-8**: LLADA with 8 codes (当前分支)

## 预期结果

### 乐观场景
```
LLADA-8 性能:
- NDCG@10: ~0.15 (略低于LLADA-32的~0.17)
- Code Validity: 0.6-0.8 (显著高于LLADA-32的0.2-0.4)
- 推理速度: 4倍于LLADA-32

结论: 可以接受的性能损失，换取显著的速度和合法率提升
```

### 悲观场景
```
LLADA-8 性能:
- NDCG@10: ~0.10 (显著低于LLADA-32)
- Code Validity: 0.4-0.5 (提升不明显)

结论: 8个codes信息量不足，需要更多codes (尝试16?)
```

## 后续实验

如果LLADA-8效果不错，可以尝试：
- **LLADA-16**: 中间方案，平衡速度和质量
- **LLADA-4**: 极致速度，1步完成推理
- **Adaptive length**: 根据任务难度动态调整codes数量

## 数据收集

重点关注：
1. **训练速度**: 每个epoch耗时
2. **推理速度**: 每个batch的评估时间
3. **Code Validity**: exact_match_rate和avg_max_match
4. **推荐质量**: NDCG, Recall
5. **Convergence**: 多少epoch达到最佳性能

## Notes

- 需要重新训练OPQ（因为n_codebook改变了）
- Semantic IDs会被重新生成
- 可以复用相同的sentence embeddings

