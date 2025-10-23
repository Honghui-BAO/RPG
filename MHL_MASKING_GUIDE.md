# MHL Model with Token Masking Guide

## 功能概述

MHL模型现在支持token masking功能，可以在训练过程中同时学习两个任务：

1. **推荐任务** - 原始的序列推荐任务
2. **重构任务** - 被mask的token的重构任务

## 核心特性

### 1. 双Loss设计
- **推荐Loss** - 正常的序列推荐损失
- **重构Loss** - 被mask的token的重构损失
- **总Loss** = 推荐Loss + 重构权重 × 重构Loss

### 2. 智能Masking策略
- 随机选择token进行masking
- 只对有效序列位置进行masking（忽略padding）
- 可配置的masking比例

### 3. 灵活的参数控制
- `mask_ratio`: 需要mask的token比例 (0.0-1.0)
- `reconstruction_weight`: 重构loss的权重
- 支持训练时动态调整

## 使用方法

### 1. 基本使用
```bash
# 使用默认masking参数
python run_mhl.py --category Beauty --gpu 0

# 自定义masking参数
python run_mhl.py --category Beauty --gpu 0 --mask_ratio 0.2 --reconstruction_weight 0.3
```

### 2. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mask_ratio` | 0.15 | 需要mask的token比例 (0.0-1.0) |
| `reconstruction_weight` | 0.5 | 重构loss的权重 |

### 3. 配置文件
在 `genrec/models/MHL/config.yaml` 中可以设置默认参数：

```yaml
# Token masking configs
mask_ratio: 0.15  # Percentage of tokens to mask (0.0 to 1.0)
reconstruction_weight: 0.5  # Weight for reconstruction loss
```

## 技术实现

### 1. Masking机制
```python
def _create_masked_tokens(self, input_tokens, attention_mask):
    # 1. 计算需要mask的位置
    # 2. 随机选择token进行mask
    # 3. 用特殊token替换被mask的token
    # 4. 返回masked tokens和mask位置
```

### 2. 双Loss计算
```python
def forward(self, batch, return_loss=True):
    # 1. 计算推荐loss（原始任务）
    recommendation_loss = self._calculate_recommendation_loss(...)
    
    # 2. 计算重构loss（masked tokens）
    reconstruction_loss = self._calculate_reconstruction_loss(...)
    
    # 3. 组合两个loss
    total_loss = recommendation_loss + self.reconstruction_weight * reconstruction_loss
```

### 3. 训练监控
模型输出包含三个loss值：
- `outputs.loss` - 总损失
- `outputs.recommendation_loss` - 推荐损失
- `outputs.reconstruction_loss` - 重构损失

## 实验建议

### 1. 参数调优
- **mask_ratio**: 建议从0.1开始，逐步调整到0.2-0.3
- **reconstruction_weight**: 建议从0.3开始，根据两个loss的平衡调整

### 2. 监控指标
- 观察两个loss的变化趋势
- 确保重构loss不会过大影响推荐性能
- 监控验证集上的推荐指标

### 3. 消融实验
```bash
# 无masking训练
python run_mhl.py --category Beauty --mask_ratio 0.0

# 不同masking比例
python run_mhl.py --category Beauty --mask_ratio 0.1
python run_mhl.py --category Beauty --mask_ratio 0.2
python run_mhl.py --category Beauty --mask_ratio 0.3

# 不同重构权重
python run_mhl.py --category Beauty --reconstruction_weight 0.3
python run_mhl.py --category Beauty --reconstruction_weight 0.5
python run_mhl.py --category Beauty --reconstruction_weight 0.7
```

## 测试验证

运行测试脚本验证功能：
```bash
python test_masking.py
```

## 注意事项

1. **内存使用**: Masking会增加内存使用，建议适当调整batch size
2. **训练时间**: 双loss计算会增加训练时间
3. **参数平衡**: 需要仔细调整两个loss的权重平衡
4. **收敛性**: 监控训练过程，确保两个任务都能收敛

## 预期效果

- **更好的表示学习**: 通过重构任务学习更丰富的item表示
- **提高泛化能力**: Masking有助于提高模型的泛化性能
- **增强鲁棒性**: 对噪声和缺失数据更加鲁棒
