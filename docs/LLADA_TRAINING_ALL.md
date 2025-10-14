# LLADA Multi-Timestep Training

## 核心思想

对每个training sample使用**多个时间步**，而不是随机采样一个。

## 当前方法 vs 新方法

### **Original（单时间步）**
```python
用户序列: [item_1, item_2, item_3]
Target: item_4

# 每个epoch只看到1个噪声水平
Epoch 1: t=15 → mask 15/32 codes
Epoch 2: t=23 → mask 23/32 codes  
Epoch 3: t=8  → mask 8/32 codes
...

# 需要很多epochs才能见到所有噪声水平
```

### **Multi-Timestep（train_timesteps_per_sample=4）**
```python
用户序列: [item_1, item_2, item_3]
Target: item_4

# 每个epoch看到4个不同噪声水平
Sample 1: t=5  → mask 5/32 codes   → loss_1
Sample 2: t=15 → mask 15/32 codes  → loss_2
Sample 3: t=22 → mask 22/32 codes  → loss_3
Sample 4: t=30 → mask 30/32 codes  → loss_4

Final loss = mean([loss_1, loss_2, loss_3, loss_4])

# 同一个target，4种不同难度的任务
```

## 实现细节

```python
# 配置
train_timesteps_per_sample: 4

# 训练时的扩展
valid_labels: [item_100, item_200]  # 2个targets

# 扩展为8个training samples
expanded: [
    item_100, item_100, item_100, item_100,  # 4个不同t
    item_200, item_200, item_200, item_200   # 4个不同t
]

# 采样8个不同的t
t: [5, 15, 22, 30, 3, 18, 25, 31]

# 分别计算8个loss，然后平均
```

## 优势分析

### 1. **更快收敛**
```python
Original: 
- 每个target每epoch只见1个噪声水平
- 需要32个epochs才能见遍所有噪声水平

Multi-timestep=4:
- 每个target每epoch见4个噪声水平
- 只需8个epochs就能见遍所有噪声水平
→ 收敛快4倍
```

### 2. **更稳定的梯度**
```python
Original:
- 梯度来自单一噪声水平
- 方差大

Multi-timestep:
- 梯度来自多个噪声水平的平均
- 方差小，更稳定
```

### 3. **更充分的训练**
```python
# 每个sample被更充分利用
- 不是只在1个噪声水平学习
- 而是在多个噪声水平同时学习
→ 数据效率更高
```

## 劣势分析

### 1. **训练时间增加**
```python
train_timesteps_per_sample=1: 1x 训练时间
train_timesteps_per_sample=4: ~4x 训练时间（每个sample forward 4次）
train_timesteps_per_sample=32: ~32x 训练时间（太慢！）
```

### 2. **显存占用增加**
```python
# Batch扩展了K倍
effective_batch_size = batch_size * train_timesteps_per_sample

# 可能需要减小实际batch size
train_batch_size: 256 → 64 (如果用4个timesteps)
```

### 3. **可能过拟合**
```python
# 每个target在一个epoch内被训练4次（不同噪声水平）
# 相当于data augmentation
# 如果数据量小，可能导致过拟合
```

## 推荐配置

### **平衡配置（推荐）**
```yaml
train_timesteps_per_sample: 4
train_batch_size: 64  # 减小以适应4倍扩展
```
- 训练时间: 原版的~1x (batch小了4倍，但每个sample用4倍)
- 收敛速度: 快4倍
- 稳定性: 更好

### **快速收敛**
```yaml
train_timesteps_per_sample: 8
train_batch_size: 32
```
- 更快收敛，但显存占用大

### **极致训练**
```yaml
train_timesteps_per_sample: 32  # 所有时间步
train_batch_size: 8
```
- 每个epoch见遍所有噪声水平
- 训练最充分，但很慢

## 实验计划

### Phase 1: 验证有效性
```bash
# 对比single vs multi timesteps
git checkout llada  # t=1
git checkout llada_training_all  # t=4

观察:
- 收敛速度（多少epoch达到最佳）
- 最终性能（NDCG@10）
- 训练时间
```

### Phase 2: 寻找最优K
```
K=1: baseline
K=2: 轻量增强
K=4: 推荐
K=8: 激进
K=16,32: 可能过度
```

### Phase 3: 结合其他优化
```
Multi-timestep + len8 + BERT backbone
→ 可能的最佳组合
```

## 预期结果

### 乐观场景
```
train_timesteps_per_sample=4:
- 5个epochs达到原版20个epochs的效果
- 最终NDCG提升2-3%
- 训练时间持平（batch小了，但效率高）
→ 明显更优
```

### 一般场景
```
- 收敛稍快（10 epochs vs 15 epochs）
- 最终性能持平
- 训练时间略增
→ 可以使用
```

### 悲观场景
```
- 过拟合严重
- 性能下降
- 训练时间增加
→ 不推荐使用
```

## 使用方法

```bash
# 训练
git checkout llada_training_all
./scripts/train_llada.sh

# 调整参数
--train_timesteps_per_sample=4  # 或2, 8, 16
--train_batch_size=64           # 相应减小batch size
```

## 关键权衡

```
训练时间 vs 样本效率

单时间步:
- 快
- 需要更多epochs
- 梯度方差大

多时间步:
- 每step慢K倍
- 需要更少epochs (快K倍)
- 梯度更稳定
→ 总体可能持平或更快
```

