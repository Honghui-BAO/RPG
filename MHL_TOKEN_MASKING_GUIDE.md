# MHL Model with Token-Level Masking Guide

## 🎯 功能概述

MHL模型现在支持**token级别的masking**，相比item级别的masking，提供了更细粒度的学习能力。

## 🔍 Token-Level vs Item-Level Masking

### Item-Level Masking (原版本)
```
原始序列: [item_A, item_B, item_C, item_D]
Item mask: [item_A, [MASK], item_C, [MASK]]
特点: 整个item被mask，需要重构所有tokens
```

### Token-Level Masking (新版本)
```
原始序列: [tokens_A, tokens_B, tokens_C, tokens_D]
Token mask: [tokens_A, [t1,t2,MASK,t4], tokens_C, [MASK,t2,t3,t4]]
特点: 部分token被mask，更细粒度的学习
```

## 🚀 使用方法

### 1. 基本使用
```bash
# 使用默认参数
./run_mhl_token_masking.sh

# 指定所有参数
./run_mhl_token_masking.sh Beauty 0 0.15 0.5

# 参数说明
# 参数1: 数据集类别
# 参数2: GPU ID  
# 参数3: Token mask比例 (0.0-1.0)
# 参数4: 重构权重
```

### 2. 测试功能
```bash
# 测试token级别masking功能
python test_token_masking.py
```

### 3. 参数扫描
```bash
# 使用Python脚本进行参数扫描
python parameter_sweep_mhl.py --category Beauty --gpu 0

# 使用快速扫描脚本
./quick_sweep_mhl.sh Beauty 0
```

## 🔧 技术实现

### 1. Token-Level Masking策略
```python
# 计算需要mask的token总数
total_valid_tokens = valid_len * n_digit
num_tokens_to_mask = max(1, int(total_valid_tokens * mask_ratio))

# 随机选择token位置进行mask
for seq_idx in range(valid_len):
    for digit_idx in range(n_digit):
        if selected_for_masking:
            mask_positions[i, seq_idx, digit_idx] = True
```

### 2. 重构损失计算
```python
# 获取所有被mask的token位置
masked_states = masked_final_states_flat[masked_positions_flat]
original_tokens_masked = original_tokens_flat[masked_positions_flat]

# 计算每个digit的重构损失
for digit in range(n_digit):
    digit_logits = masked_states[digit] @ token_embeddings[digit]
    digit_loss = CrossEntropyLoss(digit_logits, original_tokens[digit])
```

### 3. 双Loss设计
- **推荐Loss**: 正常的序列推荐任务
- **重构Loss**: 被mask的token的重构任务
- **总Loss**: 推荐Loss + 重构权重 × 重构Loss

## 📊 优势分析

### 1. 更细粒度的学习
- **Token级别**: 学习单个token的表示
- **Item级别**: 学习整个item的表示
- **效果**: Token级别提供更丰富的学习信号

### 2. 更灵活的训练
- **部分masking**: 可以只mask部分token
- **渐进学习**: 从简单到复杂的masking策略
- **适应性**: 根据数据特征调整masking策略

### 3. 更好的泛化能力
- **细粒度表示**: 学习更精细的item表示
- **鲁棒性**: 对噪声和缺失数据更鲁棒
- **多样性**: 增加训练的多样性

## 🎛️ 参数调优

### 1. Mask Ratio调优
```bash
# 测试不同的mask比例
for ratio in 0.05 0.1 0.15 0.2 0.25; do
    ./run_mhl_token_masking.sh Beauty 0 $ratio 0.5
done
```

### 2. 重构权重调优
```bash
# 测试不同的重构权重
for weight in 0.3 0.5 0.7 1.0; do
    ./run_mhl_token_masking.sh Beauty 0 0.15 $weight
done
```

### 3. 组合调优
```bash
# 使用参数扫描脚本
python parameter_sweep_mhl.py \
    --category Beauty \
    --mask_ratios 0.05 0.1 0.15 0.2 \
    --reconstruction_weights 0.3 0.5 0.7 \
    --epochs 20
```

## 📈 实验建议

### 1. 渐进式实验
```bash
# 步骤1: 测试基本功能
python test_token_masking.py

# 步骤2: 快速参数扫描
./quick_sweep_mhl.sh Beauty 0

# 步骤3: 详细参数调优
python parameter_sweep_mhl.py --category Beauty --epochs 50

# 步骤4: 结果分析
python analyze_sweep_results.py --results_dir results/quick_sweep
```

### 2. 监控指标
- **总损失**: 推荐损失 + 重构损失
- **推荐损失**: 序列推荐任务的损失
- **重构损失**: Token重构任务的损失
- **训练时间**: 每个epoch的训练时间
- **收敛性**: 损失下降的趋势

### 3. 结果分析
```bash
# 分析实验结果
python analyze_sweep_results.py \
    --results_dir results/parameter_sweep \
    --output_dir results/analysis
```

## ⚠️ 注意事项

### 1. 计算复杂度
- **Token级别**: 计算复杂度更高
- **内存使用**: 需要更多内存
- **训练时间**: 可能比item级别更长

### 2. 参数平衡
- **Mask比例**: 不宜过高，建议0.1-0.2
- **重构权重**: 需要仔细调整平衡
- **学习率**: 可能需要调整学习率

### 3. 收敛性
- **监控训练**: 确保两个loss都能收敛
- **早停策略**: 使用patience避免过拟合
- **验证集**: 定期在验证集上评估

## 🔄 与Item-Level对比

| 特性 | Item-Level | Token-Level |
|------|------------|-------------|
| 粒度 | 粗粒度 | 细粒度 |
| 计算复杂度 | 低 | 高 |
| 学习信号 | 强 | 更丰富 |
| 训练难度 | 中等 | 较高 |
| 泛化能力 | 好 | 更好 |

## 🎯 推荐使用策略

1. **首次使用**: 运行`test_token_masking.py`验证功能
2. **参数调优**: 使用`parameter_sweep_mhl.py`进行系统调优
3. **结果分析**: 使用`analyze_sweep_results.py`分析结果
4. **生产使用**: 使用最优参数进行完整训练

## 📝 日志文件

所有实验都会在`logs/`目录下生成详细日志：
- 训练过程日志
- 损失变化记录
- 性能指标记录
- 错误信息记录

现在你可以开始使用token级别的masking进行实验了！
