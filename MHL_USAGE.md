# MHL Model Usage Guide

## 脚本说明

我为你创建了三个运行脚本，方便使用MHL模型：

### 1. 完整运行脚本 (`run_mhl.sh`)
功能最全面的脚本，包含所有数据集的优化参数。

**使用方法：**
```bash
# 使用默认参数（Beauty数据集）
./run_mhl.sh

# 指定数据集
./run_mhl.sh Beauty 0
./run_mhl.sh Sports_and_Outdoors 0
./run_mhl.sh Toys_and_Games 0
./run_mhl.sh CDs_and_Vinyl 0
```

**参数：**
- 第一个参数：数据集类别
- 第二个参数：GPU ID

### 2. 快速运行脚本 (`quick_run_mhl.sh`)
简化版本，用于快速测试。

**使用方法：**
```bash
# 快速测试（使用Beauty数据集）
./quick_run_mhl.sh

# 指定数据集和GPU
./quick_run_mhl.sh Beauty 0
```

### 3. Python运行脚本 (`run_mhl.py`)
最灵活的Python脚本，支持自定义参数。

**使用方法：**
```bash
# 基本使用
python run_mhl.py --category Beauty --gpu 0

# 自定义参数
python run_mhl.py --category Beauty --gpu 0 --lr 0.005 --batch_size 16 --epochs 20

# 查看帮助
python run_mhl.py --help
```

## 数据集配置

每个数据集都有优化的默认参数：

| 数据集 | 学习率 | 温度 | 码本数 | 束搜索 | 边数 | 传播步数 |
|--------|--------|------|--------|--------|------|----------|
| Beauty | 0.01 | 0.03 | 32 | 20 | 200 | 3 |
| Sports_and_Outdoors | 0.003 | 0.03 | 16 | 100 | 30 | 5 |
| Toys_and_Games | 0.003 | 0.03 | 16 | 200 | 20 | 3 |
| CDs_and_Vinyl | 0.001 | 0.03 | 64 | 20 | 500 | 5 |

## 日志文件

所有脚本都会在 `logs/` 目录下生成日志文件，格式为：
```
logs/mhl_{category}_{timestamp}.log
```

## 推荐使用方式

1. **首次测试**：使用 `quick_run_mhl.sh` 快速验证模型能正常运行
2. **正式训练**：使用 `run_mhl.sh` 或 `run_mhl.py` 进行完整训练
3. **参数调优**：使用 `run_mhl.py` 自定义参数进行实验

## 示例命令

```bash
# 快速测试
./quick_run_mhl.sh Beauty 0

# 完整训练
./run_mhl.sh Beauty 0

# 自定义参数训练
python run_mhl.py --category Beauty --gpu 0 --lr 0.005 --batch_size 16 --epochs 30
```

## 注意事项

1. 确保已安装所有依赖：`pip install -r requirements.txt`
2. 确保有足够的GPU内存（建议至少8GB）
3. 首次运行会自动下载数据集
4. 训练时间取决于数据集大小和硬件配置
