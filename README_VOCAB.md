# RPG Vocab Tokenizer 使用指南

## 快速开始

### 一键运行
```bash
chmod +x run_train.sh
./run_train.sh
```

## 配置修改

在 `run_train.sh` 开头修改：
```bash
DATASET="Beauty"         # 数据集
VOCAB_EPOCHS=100         # vocab训练轮数
RPG_EPOCHS=50            # RPG训练轮数
BACKBONE="t5"            # t5 或 gpt2
```

## 分步运行

### 步骤1：训练Vocab Tokenizer
```bash
python simple_train_vocab.py
```

### 步骤2：训练RPG模型
```bash
python main.py --category=Beauty --tokenizer_type=vocab --backbone=t5 --n_embd=768
```

## 文件说明

- `run_train.sh` - 一键训练（推荐）
- `simple_train_vocab.py` - 简化训练脚本
- `train_vocab_tokenizer.py` - 完整训练脚本
