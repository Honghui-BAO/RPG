# LLaDA Implementation Quick Start

## ✅ 已完成的工作

我已经在 `llada` 分支上实现了基于 [LLaDA论文](https://arxiv.org/abs/2502.09992) 的diffusion推荐模型。

### 文件结构

```
RPG/
├── LLADA_DESIGN.md              # 详细设计文档
├── LLADA_QUICKSTART.md          # 本文件
├── train_llada.sh               # 训练脚本
├── test_llada.sh                # 测试脚本
└── genrec/models/LLADA/
    ├── __init__.py              # 模型导出
    ├── model.py                 # LLaDA核心实现
    ├── tokenizer.py             # 带MASK token的tokenizer
    └── config.yaml              # LLaDA配置
```

## 🎯 核心改变

### RPG (Autoregressive) vs LLaDA (Diffusion)

| 维度 | RPG | LLaDA |
|------|-----|-------|
| 建模方式 | 自回归：顺序预测下一个item | Diffusion：并行去噪所有codes |
| 训练 | 最大化 p(next\|history) | 最大化 p(clean\|noisy, t) |
| 推理 | 1次forward | T次迭代去噪 |
| 多样性 | 低 | 高 |

### 训练过程

```python
# 1. 采样时间步 t ~ Uniform(1, T)
# 2. 根据t mask掉一定比例的target codes
# 3. Transformer预测被mask的codes
# 4. 只在被mask的位置计算loss
```

### 推理过程

```python
# 从全mask状态开始
# 迭代T步，逐步去噪
# 每步保留高置信度的预测，继续mask低置信度的
# 最终得到完整的codes
```

## 🚀 使用方法

### 1. 训练LLaDA模型

```bash
./train_llada.sh
```

或者自定义参数：

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model=LLADA \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --diffusion_steps=32 \
    --mask_schedule=linear
```

### 2. 测试LLaDA模型

```bash
# 修改test_llada.sh中的checkpoint路径
./test_llada.sh
```

### 3. 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--diffusion_steps` | 32 | 扩散总步数T，越大越慢但可能更准 |
| `--mask_schedule` | linear | 遮蔽策略：linear, cosine, square |
| `--temperature` | 0.03 | softmax温度 |

## 📊 实验计划

### Phase 1: 验证基础功能
- [ ] 在Beauty数据集上训练
- [ ] 验证loss下降
- [ ] 检查生成的codes是否合理

### Phase 2: 对比实验
- [ ] RPG vs LLaDA性能对比
- [ ] 不同diffusion_steps的影响 (10, 20, 32, 50)
- [ ] 不同mask_schedule的影响 (linear, cosine, square)

### Phase 3: 高级功能
- [ ] 条件生成（固定某些codes）
- [ ] 多样性采样
- [ ] 与图搜索结合

## 🔧 调试建议

### 检查模型是否正常工作

```python
# 在Python中测试
from genrec.models.LLADA.model import LLaDARecommender
from genrec.models.LLADA.tokenizer import LLaDATokenizer

# 检查vocabulary大小
print(f"Vocab size: {tokenizer.vocab_size}")  # 应该是8195
print(f"MASK token: {tokenizer.mask_token_id}")  # 应该是8194

# 检查前向传播
batch = {...}
outputs = model.forward(batch)
print(f"Loss: {outputs.loss.item()}")
```

### 常见问题

**Q: Loss不下降？**
- 检查diffusion_steps是否太大
- 尝试调小learning rate
- 检查mask_schedule是否合适

**Q: 推理太慢？**
- 减小diffusion_steps（如从32降到16）
- 考虑使用graph-constrained decoding加速

**Q: 生成的items不合理？**
- 检查置信度阈值
- 尝试不同的mask_schedule
- 增加训练epochs

## 📝 TODO

- [ ] 实现更多mask_schedule (cosine, square)
- [ ] 添加条件生成功能
- [ ] 优化inference速度
- [ ] 添加可视化工具（diffusion过程）
- [ ] 实现多样性采样策略

## 📚 参考资料

- 论文: [LLaDA: Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- 相关工作:
  - D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces
  - Diffusion-LM: Improving Controllable Text Generation
  - DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models

## 🎉 下一步

1. 运行训练脚本验证实现
2. 对比RPG和LLaDA的性能
3. 探索diffusion在推荐系统中的优势
4. 发论文！💪

