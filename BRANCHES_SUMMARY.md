# LLADA分支总结

## 分支树
```
main
  └── llada (32 codes, baseline)
       ├── llada_len8 (8 codes, 平衡)
       └── llada_len4 (4 codes, 极速)
```

## 快速配置表

| 分支 | n_codebook | diffusion_steps | codes_per_step | mapping | vocab_size |
|------|-----------|----------------|----------------|---------|-----------|
| **llada** | 32 | 32 | 8 | count | 8195 |
| **llada_len8** | 8 | 8 | 4 | hybrid | 2051 |
| **llada_len4** | 4 | 4 | 2 | embedding | 1027 |

## 使用方法

### 训练len8 (推荐)
```bash
git checkout llada_len8
./scripts/train_llada.sh
```

### 训练len4 (极速)
```bash
git checkout llada_len4
./scripts/train_llada.sh
```

### 训练len32 (baseline)
```bash
git checkout llada
./scripts/train_llada.sh
```

## 关键差异

### llada_len8 vs llada_len4

**llada_len8特点**：
- ✅ 合理的速度（2步推理）
- ✅ 合理的质量（预期NDCG@10 ~0.15）
- ✅ 较高的code validity（40-70%）
- ✅ 使用hybrid映射（更鲁棒）

**llada_len4特点**：
- ✅ 极致速度（2步推理，更少计算）
- ⚠️ 质量可能下降（预期NDCG@10 ~0.12）
- ✅ 最高code validity（60-90%）
- ⚠️ 必须用embedding映射（count太粗糙）

## 实验建议

1. **先跑len8** - 作为主要实验
2. **再跑len4** - 验证极限
3. **对比len32** - 如果有资源的话

观察指标：
- Code Validity趋势
- NDCG/Recall
- 训练/推理时间

