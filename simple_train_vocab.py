"""
简单易懂的Vocab Tokenizer训练脚本

用法:
    python simple_train_vocab.py

这个脚本会：
1. 加载数据集
2. 训练vocab tokenizer（将item表示为4个T5 tokens）
3. 保存训练好的模型
"""

import os
import torch
import numpy as np
from tqdm import tqdm

# =============================================================================
# 配置参数 - 在这里修改您的设置
# =============================================================================
DATASET = "Beauty"              # 数据集名称（可选: Beauty, AmazonReviews2014, 等）
EPOCHS = 100                    # 训练轮数
BATCH_SIZE = 256               # 批次大小
LEARNING_RATE = 0.001          # 学习率
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

print("=" * 80)
print("Vocab Tokenizer 训练")
print("=" * 80)
print(f"数据集: {DATASET}")
print(f"训练轮数: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"学习率: {LEARNING_RATE}")
print(f"设备: {DEVICE}")
print("=" * 80)
print()

# =============================================================================
# 步骤1: 初始化模型和数据集
# =============================================================================
print("步骤1: 加载数据集和模型...")

from genrec.pipeline import Pipeline

# 创建pipeline（会自动加载数据集）
config = {
    'tokenizer_type': 'vocab',
    'vocab_epochs': EPOCHS,
    'vocab_batch_size': BATCH_SIZE,
    'vocab_lr': LEARNING_RATE,
    'device': DEVICE,
}

pipeline = Pipeline(
    model_name='RPG',
    dataset_name=DATASET,
    config_dict=config
)

tokenizer = pipeline.tokenizer
dataset = pipeline.raw_dataset  # 注意：Pipeline使用raw_dataset属性
vocab_tokenizer = tokenizer.vocab_tokenizer

print(f"✓ 数据集加载完成：{dataset.n_items - 1} 个items")
print()

# =============================================================================
# 步骤2: 加载T5 embeddings
# =============================================================================
print("步骤2: 加载T5 embeddings...")

sent_emb_path = os.path.join(
    dataset.cache_dir, 'processed',
    f'{os.path.basename(tokenizer.config["sent_emb_model"])}.sent_emb'
)

sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, 768)
sent_embs_tensor = torch.from_numpy(sent_embs).to(DEVICE)

print(f"✓ Embeddings加载完成：{sent_embs.shape}")
print()

# =============================================================================
# 步骤3: 训练Vocab Tokenizer
# =============================================================================
print("步骤3: 开始训练...")
print("=" * 80)

# 初始化优化器
optimizer = torch.optim.Adam(vocab_tokenizer.parameters(), lr=LEARNING_RATE)

# 训练
vocab_tokenizer.train()
n_items = sent_embs_tensor.shape[0]
losses = []

for epoch in range(EPOCHS):
    epoch_losses = []
    
    # 随机打乱
    perm = torch.randperm(n_items)
    
    # 进度条
    pbar = tqdm(range(0, n_items, BATCH_SIZE), desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for i in pbar:
        # 获取batch
        batch_indices = perm[i:i+BATCH_SIZE]
        batch_embs = sent_embs_tensor[batch_indices]
        
        # 前向传播
        output = vocab_tokenizer(batch_embs)
        loss = output['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录loss
        epoch_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # 计算epoch平均loss
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)
    
    print(f'Epoch {epoch+1}/{EPOCHS} - 平均Loss: {avg_loss:.6f}')

print("=" * 80)
print(f"✓ 训练完成！最终Loss: {losses[-1]:.6f}")
print()

# =============================================================================
# 步骤4: 保存模型
# =============================================================================
print("步骤4: 保存模型...")

checkpoint_path = os.path.join(
    dataset.cache_dir, 'processed',
    f'{os.path.basename(tokenizer.config["sent_emb_model"])}_vocab_tokenizer.pt'
)

torch.save({
    'model_state_dict': vocab_tokenizer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses,
    'config': {
        'vocab_epochs': EPOCHS,
        'vocab_batch_size': BATCH_SIZE,
        'vocab_lr': LEARNING_RATE,
    }
}, checkpoint_path)

print(f"✓ 模型已保存到: {checkpoint_path}")
print()

# =============================================================================
# 步骤5: 生成4-token表示
# =============================================================================
print("步骤5: 生成4-token表示...")

vocab_tokenizer.eval()
with torch.no_grad():
    token_indices = vocab_tokenizer.tokenize_items(sent_embs_tensor)

# 保存
import json
token_path = os.path.join(
    dataset.cache_dir, 'processed',
    f'{os.path.basename(tokenizer.config["sent_emb_model"])}_vocab_tokens.json'
)

item2tokens = {}
for i in range(token_indices.shape[0]):
    item = tokenizer.id2item[i + 1]
    tokens = tuple((token_indices[i] + 1).cpu().numpy().tolist())
    item2tokens[item] = tokens

with open(token_path, 'w') as f:
    json.dump(item2tokens, f)

print(f"✓ Token表示已保存到: {token_path}")
print()

# =============================================================================
# 完成
# =============================================================================
print("=" * 80)
print("训练完成！")
print("=" * 80)
print()
print("示例tokenization结果（前5个items）:")
for idx, (item, tokens) in enumerate(list(item2tokens.items())[:5]):
    print(f"  {item}: {tokens}")
print()
print("接下来可以训练RPG模型:")
print("  python main.py --model RPG --dataset Beauty --tokenizer_type vocab --backbone t5")
print()
