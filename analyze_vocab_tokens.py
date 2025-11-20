"""
分析Vocab Tokenizer的Token利用率

分析每个位置（4个位置）上使用了多少不同的tokens，以及利用率
"""

import json
import numpy as np
from collections import Counter

# =============================================================================
# 配置路径
# =============================================================================
VOCAB_TOKENS_PATH = 'cache/AmazonReviews2014/Beauty/processed/sentence-t5-base_vocab_tokens.json'
T5_VOCAB_SIZE = 32128  # T5-base词表大小

print("=" * 80)
print("Vocab Tokenizer Token利用率分析")
print("=" * 80)
print()

# =============================================================================
# 加载token数据
# =============================================================================
print(f"加载tokens from: {VOCAB_TOKENS_PATH}")
with open(VOCAB_TOKENS_PATH, 'r') as f:
    item2tokens = json.load(f)

print(f"✓ 加载了 {len(item2tokens)} 个items的token表示")
print()

# =============================================================================
# 分析每个位置的token使用情况
# =============================================================================
print("=" * 80)
print("每个位置的Token利用率分析")
print("=" * 80)
print()

# 收集每个位置的所有tokens
position_tokens = [set() for _ in range(4)]
position_token_counts = [Counter() for _ in range(4)]

for item, tokens in item2tokens.items():
    for pos in range(4):
        token_id = tokens[pos] - 1  # 减1因为存储时加了1（0用于padding）
        position_tokens[pos].add(token_id)
        position_token_counts[pos][token_id] += 1

# 显示每个位置的统计
print(f"总共有 {len(item2tokens)} 个items")
print(f"T5词表大小: {T5_VOCAB_SIZE}")
print()

for pos in range(4):
    unique_tokens = len(position_tokens[pos])
    utilization = (unique_tokens / T5_VOCAB_SIZE) * 100
    
    print(f"位置 {pos}:")
    print(f"  - 使用的不同tokens数: {unique_tokens:,}")
    print(f"  - 利用率: {utilization:.2f}%")
    print(f"  - 最常用的10个tokens:")
    
    for token_id, count in position_token_counts[pos].most_common(10):
        percentage = (count / len(item2tokens)) * 100
        print(f"      Token {token_id:5d}: 出现 {count:5d} 次 ({percentage:5.2f}%)")
    print()

# =============================================================================
# 整体统计
# =============================================================================
print("=" * 80)
print("整体统计")
print("=" * 80)
print()

# 所有位置总共使用的unique tokens
all_tokens = set()
for pos_tokens in position_tokens:
    all_tokens.update(pos_tokens)

print(f"所有位置总共使用的不同tokens: {len(all_tokens):,} / {T5_VOCAB_SIZE:,}")
print(f"总体利用率: {(len(all_tokens) / T5_VOCAB_SIZE) * 100:.2f}%")
print()

# =============================================================================
# 位置间重叠分析
# =============================================================================
print("=" * 80)
print("位置间Token重叠分析")
print("=" * 80)
print()

for i in range(4):
    for j in range(i+1, 4):
        overlap = position_tokens[i] & position_tokens[j]
        overlap_rate_i = (len(overlap) / len(position_tokens[i])) * 100
        overlap_rate_j = (len(overlap) / len(position_tokens[j])) * 100
        
        print(f"位置{i} ∩ 位置{j}:")
        print(f"  - 重叠tokens数: {len(overlap):,}")
        print(f"  - 占位置{i}的比例: {overlap_rate_i:.2f}%")
        print(f"  - 占位置{j}的比例: {overlap_rate_j:.2f}%")

print()

# =============================================================================
# 示例展示
# =============================================================================
print("=" * 80)
print("示例Token表示（前10个items）")
print("=" * 80)
print()

for idx, (item, tokens) in enumerate(list(item2tokens.items())[:10]):
    # 减1还原为实际token id
    actual_tokens = tuple(t - 1 for t in tokens)
    print(f"{item}: {actual_tokens}")

print()
print("=" * 80)
print("分析完成！")
print("=" * 80)
