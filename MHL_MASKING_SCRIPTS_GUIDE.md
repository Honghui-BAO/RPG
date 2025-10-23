# MHL Model Masking Scripts Guide

## 📁 脚本文件说明

我为你创建了完整的MHL模型masking功能脚本套件：

### 1. **`run_mhl_masking.sh`** - 主要运行脚本
功能最全面的masking训练脚本，支持所有数据集和自定义参数。

**使用方法：**
```bash
# 基本使用（默认参数）
./run_mhl_masking.sh

# 指定所有参数
./run_mhl_masking.sh Beauty 0 0.15 0.5

# 参数说明
# 参数1: 数据集类别 (Beauty, Sports_and_Outdoors, Toys_and_Games, CDs_and_Vinyl)
# 参数2: GPU ID
# 参数3: Mask比例 (0.0-1.0)
# 参数4: 重构权重
```

### 2. **`quick_test_masking.sh`** - 快速测试脚本
用于快速验证不同masking配置的效果。

**使用方法：**
```bash
# 快速测试不同masking配置
./quick_test_masking.sh Beauty 0
```

**测试内容：**
- 无masking (baseline)
- 轻度masking (10%)
- 中度masking (15%)
- 重度masking (25%)

### 3. **`batch_experiment_masking.sh`** - 批量实验脚本
自动测试多种参数组合，用于参数调优。

**使用方法：**
```bash
# 运行批量实验
./batch_experiment_masking.sh Beauty 0
```

**实验参数：**
- Mask比例: 0.0, 0.1, 0.15, 0.2, 0.25
- 重构权重: 0.0, 0.3, 0.5, 0.7
- 总共20个实验组合

### 4. **`analyze_masking_results.py`** - 结果分析脚本
分析实验结果，生成性能对比报告和可视化图表。

**使用方法：**
```bash
# 分析实验结果
python analyze_masking_results.py --results_dir results/masking_experiments --output_dir results/analysis

# 查看帮助
python analyze_masking_results.py --help
```

**输出内容：**
- 实验数据CSV文件
- 性能对比图表
- 参数敏感性分析
- 综合实验报告

## 🚀 使用流程

### 1. 快速开始
```bash
# 1. 快速测试masking功能
./quick_test_masking.sh Beauty 0

# 2. 运行单个实验
./run_mhl_masking.sh Beauty 0 0.15 0.5

# 3. 运行批量实验
./batch_experiment_masking.sh Beauty 0

# 4. 分析结果
python analyze_masking_results.py
```

### 2. 参数调优流程
```bash
# 步骤1: 快速测试找到大致范围
./quick_test_masking.sh Beauty 0

# 步骤2: 批量实验找到最优参数
./batch_experiment_masking.sh Beauty 0

# 步骤3: 分析结果
python analyze_masking_results.py

# 步骤4: 使用最优参数进行完整训练
./run_mhl_masking.sh Beauty 0 [最优mask_ratio] [最优reconstruction_weight]
```

## 📊 实验设计

### 参数组合矩阵
| Mask比例 | 重构权重 | 说明 |
|----------|----------|------|
| 0.0 | 0.0 | 基线（无masking） |
| 0.1 | 0.3 | 轻度masking |
| 0.15 | 0.5 | 中度masking |
| 0.2 | 0.7 | 重度masking |
| 0.25 | 0.7 | 极重度masking |

### 评估指标
- **总损失** - 推荐损失 + 重构权重 × 重构损失
- **推荐损失** - 原始推荐任务的损失
- **重构损失** - 被mask的token的重构损失

## 🔧 高级用法

### 1. 自定义实验参数
```bash
# 修改batch_experiment_masking.sh中的参数
MASK_RATIOS=(0.0 0.05 0.1 0.15 0.2)
RECONSTRUCTION_WEIGHTS=(0.0 0.2 0.4 0.6 0.8)
```

### 2. 添加新的数据集
```bash
# 在run_mhl_masking.sh中添加新的case
"New_Dataset")
    echo "Running New_Dataset with masking..."
    python main.py \
        --model=MHL \
        --category=New_Dataset \
        # ... 其他参数
        ;;
```

### 3. 自定义分析
```python
# 修改analyze_masking_results.py
# 添加新的分析指标和可视化
```

## 📈 结果解读

### 1. 性能对比
- **最佳配置** - 总损失最低的参数组合
- **参数敏感性** - 不同参数对性能的影响
- **收敛性** - 训练过程的稳定性

### 2. 可视化图表
- **损失对比图** - 不同masking配置的损失对比
- **参数热力图** - 参数组合的性能热力图
- **相关性分析** - 参数间的相关性分析

### 3. 实验报告
- **实验摘要** - 实验设置和结果概览
- **最佳配置** - 推荐的最优参数
- **性能分析** - 详细的性能分析

## ⚠️ 注意事项

1. **资源需求** - Masking会增加内存和计算需求
2. **训练时间** - 批量实验需要较长时间
3. **参数平衡** - 需要仔细调整两个loss的权重
4. **收敛性** - 监控训练过程确保收敛

## 🎯 推荐使用策略

1. **首次使用** - 运行`quick_test_masking.sh`了解基本效果
2. **参数调优** - 使用`batch_experiment_masking.sh`进行系统调优
3. **结果分析** - 使用`analyze_masking_results.py`分析结果
4. **最终训练** - 使用最优参数进行完整训练

## 📝 日志文件

所有脚本都会在`logs/`目录下生成详细的日志文件：
- 训练过程日志
- 损失变化记录
- 性能指标记录
- 错误信息记录

## 🔍 故障排除

1. **内存不足** - 减小batch_size
2. **训练不收敛** - 调整学习率或权重
3. **参数冲突** - 检查参数设置是否合理
4. **日志分析** - 查看详细日志定位问题
