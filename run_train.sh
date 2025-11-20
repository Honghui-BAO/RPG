#!/bin/bash

# =============================================================================
# RPG + Vocab Tokenizer 一键训练脚本
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数 - 在这里修改您的设置
# =============================================================================
DATASET="Beauty"                # 数据集名称
DEVICE="cuda:0"                 # 设备

# Vocab Tokenizer 参数
VOCAB_EPOCHS=100                # vocab训练轮数
VOCAB_BATCH_SIZE=256            # vocab batch size
VOCAB_LR=0.001                  # vocab学习率

# RPG 模型参数
RPG_EPOCHS=50                   # RPG训练轮数
RPG_LR=0.01                     # RPG学习率
RPG_BATCH_SIZE=512              # RPG batch size
BACKBONE="t5"                   # backbone类型: t5 或 gpt2
TEMPERATURE=0.03                # temperature

echo "========================================="
echo "  RPG + Vocab Tokenizer 训练"
echo "========================================="
echo "数据集: $DATASET"
echo "设备: $DEVICE"
echo "========================================="
echo ""

# =============================================================================
# 步骤1: 训练Vocab Tokenizer
# =============================================================================
echo "========================================="
echo "步骤 1/2: 训练 Vocab Tokenizer"
echo "========================================="

python3 train_vocab_tokenizer.py \
    --model RPG \
    --dataset $DATASET \
    --vocab_epochs $VOCAB_EPOCHS \
    --vocab_batch_size $VOCAB_BATCH_SIZE \
    --vocab_lr $VOCAB_LR \
    --device $DEVICE

echo ""
echo "✅ Vocab Tokenizer训练完成！"
echo ""

# =============================================================================
# 步骤2: 训练RPG模型
# =============================================================================
echo "========================================="
echo "步骤 2/2: 训练 RPG 模型"
echo "========================================="

if [ "$BACKBONE" = "t5" ]; then
    python3 main.py \
        --category=$DATASET \
        --tokenizer_type=vocab \
        --backbone=t5 \
        --n_embd=768 \
        --n_epochs=$RPG_EPOCHS \
        --lr=$RPG_LR \
        --train_batch_size=$RPG_BATCH_SIZE \
        --temperature=$TEMPERATURE
else
    python3 main.py \
        --category=$DATASET \
        --tokenizer_type=vocab \
        --backbone=gpt2 \
        --n_epochs=$RPG_EPOCHS \
        --lr=$RPG_LR \
        --train_batch_size=$RPG_BATCH_SIZE \
        --temperature=$TEMPERATURE
fi

echo ""
echo "========================================="
echo "✅ 所有训练完成！"
echo "========================================="
