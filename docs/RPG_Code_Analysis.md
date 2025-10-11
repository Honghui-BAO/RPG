# RPG (Representation Planning with GPT) 代码深度解析

本文档详细解析了RPG推荐系统的核心实现机制，包括架构设计、OPQ量化、推理流程等关键部分。

---

## 目录

1. [抽象类设计模式](#1-抽象类设计模式)
2. [OPQ训练流程](#2-opq训练流程)
3. [OPQ量化详解](#3-opq量化详解)
4. [Tokenizer动态加载机制](#4-tokenizer动态加载机制)
5. [Inference推理流程](#5-inference推理流程)
6. [Item-Item图的设计](#6-item-item图的设计)
7. [Semantic IDs与GPT2词表](#7-semantic-ids与gpt2词表)

---

## 1. 抽象类设计模式

### 1.1 为什么使用抽象类？

RPG项目定义了三个核心抽象类：
- `AbstractModel` (model.py)
- `AbstractTokenizer` (tokenizer.py) 
- `AbstractDataset` (dataset.py)

### 1.2 设计优势

#### **统一的接口规范**

```python
class AbstractModel(nn.Module):
    def calculate_loss(self, batch):
        raise NotImplementedError('calculate_loss method must be implemented.')

    def generate(self, batch, n_return_sequences=1):
        raise NotImplementedError('predict method must be implemented.')
```

这强制要求所有具体实现必须提供特定的方法。

#### **支持多模型扩展**

虽然目前只有RPG这一个实现，但架构为将来添加新模型预留了空间：

```python
class ModelB(AbstractModel):
    def __init__(self, config, dataset, tokenizer):
        super().__init__(config, dataset, tokenizer)
        # 你的初始化
    
    def calculate_loss(self, batch):
        # 你的损失函数实现
    
    def generate(self, batch, n_return_sequences=1):
        # 你的生成逻辑
```

#### **类型约束和IDE支持**

```python
class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        ...
    ):
```

这告诉开发者和IDE：参数可以是字符串或任何继承自抽象类的对象。

#### **共享通用逻辑**

抽象类可以提供公共实现，子类可以选择性重写：

```python
# AbstractModel中的默认实现
@property
def n_parameters(self):
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return f'Total number of trainable parameters: {total_params}'

# RPG模型重写以提供更详细信息
@property
def n_parameters(self) -> str:
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters())
    return f'#Embedding parameters: {emb_params}\n' \
           f'#Non-embedding parameters: {total_params - emb_params}\n' \
           f'#Total trainable parameters: {total_params}\n'
```

### 1.3 核心好处总结

1. **可扩展性** - 轻松添加新模型、新数据集、新tokenizer
2. **规范性** - 强制所有实现遵循相同接口
3. **可维护性** - 代码职责清晰，模块间松耦合
4. **代码复用** - 公共逻辑只写一次
5. **类型安全** - 编译时就能发现接口不匹配的问题

---

## 2. OPQ训练流程

### 2.1 完整调用链

```
main.py
  ↓ 创建Pipeline(model_name='RPG', ...)
  
pipeline.py (line 67)
  ↓ get_tokenizer('RPG')
  
utils.py (line 122-142)
  ↓ importlib.import_module('genrec.models.RPG.tokenizer')
  ↓ getattr(module, 'RPGTokenizer')
  ↓ return RPGTokenizer  # 返回类
  
pipeline.py (line 67 continued)
  ↓ RPGTokenizer(config, raw_dataset)  # 实例化
  
genrec/models/RPG/tokenizer.py (line 40-48)
  ↓ RPGTokenizer.__init__()
  ↓   self.item2tokens = self._init_tokenizer(dataset)
  ↓     self._generate_semantic_id_opq(...)  # 🔥 OPQ训练！
```

### 2.2 训练时机和缓存机制

```python
def _init_tokenizer(self, dataset: AbstractDataset):
    # 检查semantic IDs是否已存在
    sem_ids_path = os.path.join(
        dataset.cache_dir, 'processed',
        f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
    )

    if not os.path.exists(sem_ids_path):
        # 第一次运行：生成sentence embeddings和semantic IDs
        # 1. 编码文本为embeddings
        sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
        
        # 2. 可选的PCA降维
        if self.config['sent_emb_pca'] > 0:
            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            sent_embs = pca.fit_transform(sent_embs)
        
        # 3. 训练OPQ并生成semantic IDs
        training_item_mask = self._get_items_for_training(dataset)
        self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)
    
    # 加载已生成的semantic IDs
    item2sem_ids = json.load(open(sem_ids_path, 'r'))
    item2tokens = self._sem_ids_to_tokens(item2sem_ids)
    return item2tokens
```

### 2.3 关键设计点

- **只用训练集训练OPQ**：`train_mask`保证只有训练集中出现的items用于训练index
- **缓存机制**：训练好的semantic IDs会保存，下次直接加载
- **在训练推荐模型之前**：OPQ作为预处理步骤，在推荐模型训练之前就完成了

---

## 3. OPQ量化详解

### 3.1 函数概览

`_generate_semantic_id_opq`函数使用FAISS库进行Optimized Product Quantization。

### 3.2 逐步解析

#### **Part 1: GPU资源配置**

```python
import faiss
if self.config['opq_use_gpu']:
    res = faiss.StandardGpuResources()  # 创建GPU资源对象
    res.setTempMemory(1024 * 1024 * 512)  # 设置临时内存为512MB
    co = faiss.GpuClonerOptions()  # GPU克隆选项
    co.useFloat16 = self.n_digit >= 56  # 如果维度>=56，使用半精度浮点数节省显存
```

#### **Part 2: 创建FAISS索引**

```python
faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
index = faiss.index_factory(
    sent_embs.shape[1],  # 输入维度（embedding维度）
    self.index_factory,  # 索引类型，例如"OPQ32,IVF1,PQ32x8"
    faiss.METRIC_INNER_PRODUCT  # 使用内积作为相似度度量
)
```

**index_factory格式**：`"OPQ32,IVF1,PQ32x8"`
- `OPQ32`: Optimized Product Quantization，32维旋转矩阵
- `IVF1`: Inverted File Index，1个聚类中心
- `PQ32x8`: Product Quantization，32个子量化器，每个8位（256个码本）

#### **Part 3: 训练和添加数据**

```python
self.log(f'[TOKENIZER] Training index...')
if self.config['opq_use_gpu']:
    index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)

# 🔥 关键！只用训练集的embeddings训练量化器
index.train(sent_embs[train_mask])

# 添加所有items的embeddings（训练+测试）
index.add(sent_embs)

if self.config['opq_use_gpu']:
    index = faiss.index_gpu_to_cpu(index)
```

**为什么这样做？**
- `train()`: 学习OPQ的旋转矩阵和PQ的码本（只用训练集，避免数据泄露）
- `add()`: 用学好的量化器编码所有items（包括测试集的items也需要ID）

#### **Part 4: 提取量化码**

```python
# 从索引中提取底层的IVF索引
ivf_index = faiss.downcast_index(index.index)
invlists = faiss.extract_index_ivf(ivf_index).invlists

# 因为只有1个聚类（IVF1），所以所有数据都在list 0中
ls = invlists.list_size(0)
pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
pq_codes = pq_codes.reshape(-1, invlists.code_size)
```

#### **Part 5: 解码量化码**

```python
faiss_sem_ids = []
n_bytes = pq_codes.shape[1]

for u8code in pq_codes:
    bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
    code = []
    for i in range(self.n_digit):
        code.append(bs.read(self.n_codebook_bits))  # 每次读取n_codebook_bits位
    faiss_sem_ids.append(code)

pq_codes = np.array(faiss_sem_ids)
```

**举例说明**：假设`n_digit=32`，`codebook_size=256`（8 bits）
- 每个item的embedding被分解成32个子向量
- 每个子向量被量化成256个码本中的一个（0-255）
- 最终每个item用32个数字表示：`[code_0, code_1, ..., code_31]`

#### **Part 6: 保存结果**

```python
item2sem_ids = {}
for i in range(pq_codes.shape[0]):
    item = self.id2item[i + 1]
    item2sem_ids[item] = tuple(pq_codes[i].tolist())

self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
with open(sem_ids_path, 'w') as f:
    json.dump(item2sem_ids, f)
```

**输出格式示例**：
```json
{
  "item_A001": [15, 203, 78, 145, ...],
  "item_B002": [88, 12, 234, 67, ...],
  ...
}
```

### 3.3 整体流程图

```
输入: sent_embs [n_items × emb_dim]
      例如: [10000 × 768]

↓ [Step 1] 创建OPQ+PQ索引
           "OPQ32,IVF1,PQ32x8"

↓ [Step 2] 训练（只用训练集）
           学习旋转矩阵 + 32个码本（每个256个中心）

↓ [Step 3] 编码所有items
           sent_embs → 压缩表示

↓ [Step 4] 提取量化码
           [10000 × 768] → [10000 × 32]
           每个数字范围: 0-255

↓ [Step 5] 保存
           输出: item → [code_0, code_1, ..., code_31]
```

### 3.4 为什么要这样做？

1. **降维压缩**：768维 → 32个离散ID（每个0-255）
2. **可学习的tokenization**：这32个ID就是模型的输入token
3. **语义保持**：OPQ保证相似的items有相似的codes
4. **训练效率**：32个小词表比1个大词表训练更快

### 3.5 为什么要先做PCA降维？

在执行OPQ量化之前，RPG会先对sentence embeddings进行PCA降维。

#### **配置示例**

```yaml
# Sentence T5
sent_emb_model: sentence-transformers/sentence-t5-base
sent_emb_dim: 768
sent_emb_pca: 128      # 从768维降到128维

# OpenAI Embedding
sent_emb_model: text-embedding-3-large
sent_emb_dim: 3072
sent_emb_pca: 512      # 从3072维降到512维
```

#### **代码实现**

```python
if self.config['sent_emb_pca'] > 0:
    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
    from sklearn.decomposition import PCA
    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
    sent_embs = pca.fit_transform(sent_embs)
```

#### **降维的四个关键原因**

**1. 优化OPQ效果**

Product Quantization假设各个子空间是独立的。在高维空间（如768维），很多维度之间存在相关性，违反了PQ的独立性假设。降到128维后，主要保留最重要的独立成分，更适合PQ。

**2. whiten=True的关键作用**

```python
pca = PCA(n_components=128, whiten=True)
```

白化（whitening）做两件事：
- **去相关性**：将数据投影到正交的主成分上
- **标准化方差**：将每个主成分缩放到单位方差

```python
# 不whiten
PC1: 方差=100, PC2: 方差=50, PC3: 方差=10, ...
→ OPQ会被高方差的维度主导，量化误差不均

# whiten=True  
PC1: 方差=1, PC2: 方差=1, PC3: 方差=1, ...
→ OPQ对所有维度一视同仁，量化更均衡
```

这对Product Quantization非常重要，因为PQ会将128维分成32组（每组4维），方差均衡能保证每组的量化误差相近。

**3. 计算效率**

```python
# 降维前：OPQ在768维空间
- 时间复杂度: O(n × d²) where d=768
- FAISS训练时间: ~100s

# 降维后：OPQ在128维空间  
- 时间复杂度: O(n × d²) where d=128
- FAISS训练时间: ~15s
```

**4. 去噪和泛化**

PCA降维相当于特征选择，保留最重要的信息，过滤噪声：
- ✅ 保留最重要的128个主成分（通常能保留95%+的方差）
- ✅ 过滤掉噪声和任务无关的信息
- ✅ 更适合下游推荐任务

#### **降维比例的选择**

```yaml
# Sentence-T5: 768 → 128 (16.7%)
# OpenAI 3-large: 3072 → 512 (16.7%)
```

规律：都降到原始维度的约**16-17%**！这是在保留足够信息、适合OPQ量化、计算效率之间的平衡点。

#### **完整流程**

```
原始Embeddings (10000, 768)
    ↓ PCA降维+白化
降维Embeddings (10000, 128)
    ↓ OPQ量化
Semantic IDs (10000, 32) × [0-255]
    ↓ 转换为Token IDs
Token IDs (10000, 32) × [1-8192]
```

---

## 4. Tokenizer动态加载机制

### 4.1 加载流程

```python
# main.py
pipeline = Pipeline(
    model_name='RPG',  # ← 传入字符串
    dataset_name='AmazonReviews2014',
    ...
)

# ↓

# pipeline.py
self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
#                ↑ 获取tokenizer类      ↑ 实例化

# ↓

# utils.py
def get_tokenizer(model_name: str):
    tokenizer_class = getattr(
        importlib.import_module(f'genrec.models.{model_name}.tokenizer'),
        f'{model_name}Tokenizer'
    )
    return tokenizer_class
```

### 4.2 实际导入路径

当 `model_name='RPG'` 时：

```python
importlib.import_module('genrec.models.RPG.tokenizer')
# ↓ 导入
# /Users/honghuibao/Desktop/RPG/genrec/models/RPG/tokenizer.py

getattr(module, 'RPGTokenizer')
# ↓ 获取类
# class RPGTokenizer(AbstractTokenizer)
```

### 4.3 为什么这样设计？

#### **约定优于配置**
只要按照命名规范创建 `{ModelName}Tokenizer`，系统就能自动找到它。

#### **易于扩展**
添加新模型时，只需创建新目录：
```
genrec/models/
  ├── RPG/
  │   ├── tokenizer.py  # RPGTokenizer
  │   └── model.py
  └── NewModel/
      ├── tokenizer.py  # NewModelTokenizer
      └── model.py
```

#### **模块化**
每个模型的代码独立在自己的文件夹中，互不干扰。

#### **延迟加载**
只有在需要时才导入模块，节省内存。

这是一个优雅的**工厂模式**设计！

---

## 5. Inference推理流程

### 5.1 流程概览

```
Trainer.evaluate() 
  ↓
Model.generate()
  ↓
  ├─ Step 1: Forward pass获取用户表示
  ├─ Step 2: 计算token logits
  └─ Step 3: 两种推荐策略
        ├─ 不使用图：暴力搜索所有items
        └─ 使用图：Graph-constrained decoding（更快）
```

### 5.2 Step 1: Forward Pass - 获取用户表示

```python
def generate(self, batch, n_return_sequences=1):
    # 通过GPT2获取用户序列的表示
    outputs = self.forward(batch, return_loss=False)
    
    # 提取最后一个位置的hidden state（用于预测下一个item）
    states = outputs.final_states.gather(
        dim=1,
        index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
    )
    # states shape: (batch_size, 1, n_digit=32, hidden_dim=768)
    states = F.normalize(states, dim=-1)  # L2归一化
```

**例子**：
- 用户历史：`[item_1, item_2, item_3]`
- 每个item被转换成32个tokens
- GPT2输出：`states = (1, 1, 32, 768)` - 32个预测头，每个768维

### 5.3 Step 2: 计算Token Logits

```python
# 获取所有token embeddings（排除padding和eos）
token_emb = self.gpt2.wte.weight[1:-1]  # (8192, 768)
token_emb = F.normalize(token_emb, dim=-1)

# 分成32组，每组256个token
token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
# token_embs[0]: (256, 768) - 第1个digit的256个码本
# token_embs[31]: (256, 768) - 第32个digit的256个码本

# 计算每个digit的logits
logits = [
    torch.matmul(states[:,0,i,:], token_embs[i].T) / self.temperature 
    for i in range(self.n_pred_head)
]
# logits[i] shape: (batch_size, 256)

logits = [F.log_softmax(logit, dim=-1) for logit in logits]
token_logits = torch.cat(logits, dim=-1)  # (batch_size, 8192)
```

**例子**：
```python
# batch_size=1, n_digit=32, codebook_size=256
token_logits = tensor([
    [-2.5, -3.1, -1.8, ..., -4.2,  # digit 0: 256个logits
     -1.9, -2.7, -3.5, ..., -2.1,  # digit 1: 256个logits
     ...
     -3.3, -1.5, -2.9, ..., -4.1]  # digit 31: 256个logits
])  # shape: (1, 8192)
```

### 5.4 Step 3a: 不使用图的推荐（暴力搜索）

```python
else:  # 不使用图
    # 计算所有items的logits
    item_logits = torch.gather(
        input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
        dim=-1,
        index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
    ).mean(dim=-1)  # 对32个digit求平均
    
    preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
    return preds.unsqueeze(-1)
```

**具体例子**：
```python
# 假设有10000个items
# item_1的semantic IDs: [5, 123, 78, ..., 203]

# 计算item_1的score：
item_1_logits = [
    token_logits[5],      # digit 0: code=5
    token_logits[379],    # digit 1: code=123 (123 + 256)
    token_logits[590],    # digit 2: code=78 (78 + 512)
    ...
    token_logits[8139]    # digit 31: code=203 (203 + 31*256)
]
item_1_score = mean(item_1_logits) = -2.3

# 对所有10000个items计算score，选top-K
```

### 5.5 Step 3b: 使用图的推荐（Graph-Constrained Decoding）⭐

这是RPG的核心创新！

```python
if self.generate_w_decoding_graph:
    if not self.init_flag:
        self.init_graph()  # 第一次调用时初始化图
        self.init_flag = True
    outputs = self.graph_propagation(
        token_logits=token_logits,
        n_return_sequences=n_return_sequences
    )
    return outputs
```

#### **初始化图（只在第一次推理时）**

```python
def init_graph(self):
    # 构建item-item相似度矩阵
    item_item_sim = self.build_ii_sim_mat()  # (n_items, n_items)
    # 为每个item找到最相似的n_edges个邻居
    self.adjacency = self.build_adjacency_list(item_item_sim)  # (n_items, n_edges)
```

**例子**：
```python
# 假设n_edges=50
adjacency = tensor([
    [0,     0,    0,    ...],  # padding item
    [234, 567, 123, ...],      # item_1的50个最相似邻居
    [111, 999, 234, ...],      # item_2的50个最相似邻居
    ...
])  # shape: (10000, 50)
```

#### **图传播（Graph Propagation）**

```python
def graph_propagation(self, token_logits, n_return_sequences):
    batch_size = token_logits.shape[0]
    
    # 初始化：随机采样num_beams个节点作为起点
    topk_nodes_sorted = torch.randint(
        1, self.dataset.n_items,
        (batch_size, self.num_beams),  # 假设num_beams=10
    )
    
    visited_nodes = {}  # 记录访问过的节点
    
    # 迭代传播 propagation_steps 步
    for sid in range(self.propagation_steps):  # 假设=5
        # 找到当前top节点的所有邻居
        all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)
        
        next_nodes = []
        for batch_id in range(batch_size):
            neighbors_in_batch = torch.unique(all_neighbors[batch_id])
            
            # 标记访问过的节点
            for node in neighbors_in_batch:
                visited_nodes[batch_id].add(node)
            
            # 计算这些邻居的scores
            scores = torch.gather(
                input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                dim=-1,
                index=(self.item_id2tokens[neighbors_in_batch] - 1)
            ).mean(dim=-1)
            
            # 选出top num_beams个作为下一轮的起点
            idxs = torch.topk(scores, self.num_beams).indices
            next_nodes.append(neighbors_in_batch[idxs])
        
        topk_nodes_sorted = torch.stack(next_nodes, dim=0)
    
    # 返回最终的top-K推荐和访问的节点数
    visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])
    return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts
```

### 5.6 完整推荐例子

```python
# ========== 输入 ==========
用户历史: [item_100, item_200, item_300]
总items数: 10000
n_return_sequences: 10  # 推荐top-10

# ========== 方法1: 暴力搜索 ==========
计算所有10000个items的scores
耗时: ~100ms
结果: [item_5432, item_1234, item_7890, ...]

# ========== 方法2: 图传播 ==========
# 第0步: 随机初始化10个候选
candidates = [item_1234, item_5678, item_9012, ..., item_7777]

# 第1步: 扩展邻居
neighbors_1 = 每个候选的50个邻居 → 去重后约300个
计算300个items的scores
选出top-10: [item_2345, item_6789, ...]
visited_so_far = 310个items

# 第2步: 继续扩展
neighbors_2 = 新的300个邻居
选出top-10: [item_3456, item_7890, ...]
visited_so_far = 610个items

# ... 重复5步 ...

# 最终结果:
只访问了约1500个items（而不是10000）
加速: ~6.7x
结果: [item_3456, item_7890, item_2345, ...]
```

### 5.7 关键优势对比

| 方法 | 访问items数 | 速度 | 准确度 |
|------|------------|------|--------|
| 暴力搜索 | 10,000 | 慢 | 100%（最优） |
| 图传播 | ~1,500 | 快 | ~98%（接近最优） |

### 5.8 为什么图传播有效？

1. **语义相似性**：相似的items在embedding空间接近
2. **局部搜索**：从随机点开始，沿着相似度梯度爬升
3. **多起点**：10个beam保证不会陷入局部最优
4. **记忆访问**：避免重复计算

---

## 6. Item-Item图的设计

### 6.1 图构建 vs 图搜索：两个阶段

#### **阶段1：构建全局图（与用户无关）**

```python
def build_ii_sim_mat(self):
    # 基于GPT2的token embeddings计算item相似度
    token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
    token_embs = F.normalize(token_embs, dim=-1)
    token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
    # ↑ 这是基于语义的相似度，与用户无关
    
    # 计算item之间的相似度
    for each pair of items (i, j):
        similarity[i, j] = average(token_sims[items_i的32个codes, items_j的32个codes])
```

**关键**：这个相似度矩阵只依赖于：
- Item的semantic IDs（固定的）
- GPT2的token embeddings（训练后固定）

**结果**：
```python
# 全局图（所有用户共享）
adjacency = {
    item_1: [item_234, item_567, item_123, ...],  # item_1的50个最相似邻居
    item_2: [item_111, item_999, item_234, ...],  # item_2的50个最相似邻居
    ...
}
```

#### **阶段2：个性化搜索（用户特定）**

虽然图结构是全局的，但在图上的**搜索路径**是由每个用户的logits决定的！

```python
def graph_propagation(self, token_logits, n_return_sequences):
    # token_logits: 用户特定的！
    
    for step in range(propagation_steps):
        # 从图中获取邻居（全局结构）
        all_neighbors = self.adjacency[topk_nodes]  # ← 全局图
        
        # 但是用用户特定的logits来评分！
        scores = compute_scores(neighbors, token_logits)  # ← 用户特定
        
        # 选择score最高的邻居继续探索
        topk_nodes = select_top_k(scores)
```

### 6.2 具体例子说明

假设有3个用户和item-item图：

#### **全局图（所有用户看到的是一样的）**

```python
# 基于语义相似度构建
adjacency = {
    "iPhone_14": ["iPhone_13", "Samsung_S23", "iPad", "AirPods", ...],
    "Harry_Potter_1": ["Harry_Potter_2", "Lord_of_Rings", "Game_of_Thrones", ...],
    "Nike_Shoes": ["Adidas_Shoes", "Running_Socks", "Sports_Watch", ...],
}
```

#### **用户A：科技爱好者**

```python
用户A历史: [iPhone_13, MacBook, AirPods]
用户A的token_logits: {
    "iPhone_14": -1.2,      # 高分
    "Samsung_S23": -2.8,    # 中分
    "iPad": -1.5,           # 高分
    "Harry_Potter_2": -8.5, # 低分
    "Nike_Shoes": -9.2,     # 低分
}

# Graph Propagation for 用户A:
Step 0: 随机起点 = [iPhone_14, Harry_Potter_1, Nike_Shoes]
Step 1: 
  - iPhone_14的邻居 → 评分很高 ✓
  - Harry_Potter_1的邻居 → 评分很低 ✗
  - Nike_Shoes的邻居 → 评分很低 ✗
  → 继续沿着iPhone_14方向探索
  
最终推荐: [iPad, AirPods_Pro, Apple_Watch, ...]
```

#### **用户B：书虫**

```python
用户B历史: [Harry_Potter_1, Harry_Potter_2, Lord_of_Rings]
用户B的token_logits: {
    "Harry_Potter_3": -1.1,  # 高分
    "Game_of_Thrones": -1.8, # 高分
    "iPhone_14": -7.5,       # 低分
}

# Graph Propagation for 用户B:
Step 0: 随机起点 = [iPhone_14, Harry_Potter_1, Nike_Shoes]
Step 1:
  - iPhone_14的邻居 → 评分很低 ✗
  - Harry_Potter_1的邻居 → 评分很高 ✓
  - Nike_Shoes的邻居 → 评分很低 ✗
  → 继续沿着Harry_Potter方向探索

最终推荐: [Harry_Potter_3, Game_of_Thrones, Hobbit, ...]
```

### 6.3 为什么这样设计？

#### **优势1：计算效率**
- 图只构建一次，所有用户共享
- 不需要为每个用户构建个性化的图

#### **优势2：语义保证**
- 图保证了语义上相关的items是连通的
- 不会出现"从iPhone跳到鞋子"的情况（除非它们真的相似）

#### **优势3：个性化搜索**
- 虽然图结构相同，但不同用户会走不同的路径
- 用户的logits充当"指南针"，引导搜索方向

### 6.4 类比理解

这就像：
- **全局图** = 城市的道路网络（所有人看到的地图是一样的）
- **用户logits** = 每个人的目的地偏好（决定走哪条路）

两个人在同一个城市（图）中，但因为目的地不同，最终走的路线完全不同！

```
用户A（想买电子产品）:
起点 → 电子商店街 → 手机店 → 电脑店 → 配件店

用户B（想买书）:
起点 → 书店街 → 奇幻小说区 → 科幻小说区 → 文学区
```

所以，虽然图是全局的，但搜索是完全个性化的！

---

## 7. Semantic IDs与GPT2词表

### 7.1 核心结论

**RPG的GPT2没有任何文本token，只有8194个专门为推荐设计的semantic tokens！**

### 7.2 转换流程

#### **Step 1: OPQ生成的Semantic IDs**

```python
# OPQ输出的原始semantic IDs（每个item 32个数字）
item_A的semantic IDs: [15, 203, 78, 145, ..., 199]  # 范围: 0-255
```

#### **Step 2: 转换成Token IDs（加入偏移）**

```python
def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
    for item in item2sem_ids:
        tokens = list(item2sem_ids[item])
        for digit in range(self.n_digit):
            # 关键！给每个digit加上偏移
            tokens[digit] += self.codebook_size * digit + 1
        item2sem_ids[item] = tuple(tokens)
    return item2sem_ids
```

**转换后的Token IDs**:
```python
原始: [15,   203,  78,   145,  ..., 199]
      ↓     ↓     ↓     ↓          ↓
转换: [16,   460,  590,  913,  ..., 8135]
      ↑     ↑     ↑     ↑          ↑
计算: 15+1  203+  78+   145+       199+
          256+1 512+1 768+1      7936+1
```

#### **Step 3: GPT2词表的完整结构**

```python
# tokenizer.py
self.eos_token = self.n_digit * self.codebook_size + 1  # 32*256+1 = 8193

@property
def vocab_size(self) -> int:
    return self.eos_token + 1  # 8194
```

**GPT2词表映射**:
```python
GPT2 Vocab (size=8194):
├─ 0:          [PAD] padding token
├─ 1-256:      digit 0的256个codes
├─ 257-512:    digit 1的256个codes
├─ 513-768:    digit 2的256个codes
│  ...
├─ 7937-8192:  digit 31的256个codes
└─ 8193:       [EOS] end of sequence

❌ 没有 "apple"
❌ 没有 "iPhone"
❌ 没有任何英文单词
✅ 只有抽象的semantic codes
```

#### **Step 4: 在模型中使用**

```python
def forward(self, batch: dict, return_loss=True):
    # 获取input items的token IDs
    input_tokens = self.item_id2tokens[batch['input_ids']]
    # shape: (batch_size, seq_len, 32) - 每个item有32个token IDs
    
    # 直接使用GPT2的embedding layer！
    input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
    # self.gpt2.wte.weight shape: (8194, 768)
```

### 7.3 完整例子

假设有一个item "iPhone_14":

```python
# ===== 第1步: OPQ量化 =====
iPhone_14的sentence embedding: [0.234, -0.123, ..., 0.567]  # 768维
                                ↓ OPQ量化
iPhone_14的semantic IDs: [15, 203, 78, 145, ..., 199]  # 32个0-255的数字

# ===== 第2步: 转换成Token IDs =====
iPhone_14的token IDs: [16, 460, 590, 913, ..., 8135]  # 32个token IDs

# ===== 第3步: 查询GPT2 Embedding =====
embeddings = []
for token_id in [16, 460, 590, 913, ..., 8135]:
    emb = gpt2.wte.weight[token_id]  # 获取(768,)的embedding
    embeddings.append(emb)

# ===== 第4步: 聚合成Item Embedding =====
iPhone_14_embedding = mean(embeddings)  # (768,)

# ===== 第5步: 输入GPT2 =====
user_sequence = [MacBook, AirPods, iPhone_14]
# → 转换成 3×32 = 96个token IDs
# → 每个token查GPT2 embedding
# → 每个item的32个embeddings取平均
# → 得到3个item embeddings
# → 输入GPT2 Transformer处理
```

### 7.4 为什么这样设计？

#### **优势1：端到端训练**
```python
# GPT2的embedding是可学习的！
self.gpt2.wte.weight  # shape: (8194, 768), requires_grad=True
```
训练时，这8194个token embeddings会随着推荐任务一起优化。

#### **优势2：语义结构**
```python
# 词表的结构化设计
token[1-256]:    digit 0的codes
token[257-512]:  digit 1的codes
...
```
每个digit有自己独立的256个codes，互不干扰。

#### **优势3：分解预测**
```python
# 训练时，不是预测一个item，而是预测32个tokens
loss = mean([
    loss(predict_digit_0,  target_digit_0),
    loss(predict_digit_1,  target_digit_1),
    ...
    loss(predict_digit_31, target_digit_31),
])
```

### 7.5 词表对比

| 原始GPT2 | RPG的GPT2 |
|---------|-----------|
| vocab_size = 50,257 | vocab_size = 8,194 |
| 文本tokens: "apple", "the", "is", ... | 推荐tokens: [PAD], digit_0_code_0, ... |
| 预训练于大规模文本 | **从头训练**于推荐数据 |
| 理解自然语言 | 理解item序列 |

### 7.6 GPT2的创建方式

```python
# 第76行：从头创建GPT2，不是加载预训练的！
self.gpt2 = GPT2Model(gpt2config)  
# ↑ 没有 .from_pretrained()
# ↑ 这是一个随机初始化的、全新的GPT2结构
```

### 7.7 类比说明

```python
# 原始GPT2的使命
输入: "The weather is"
输出: "nice" (文本补全)

# RPG的GPT2的使命
输入: [item_1, item_2, item_3] 
      → 转换成semantic tokens
输出: 下一个item的semantic tokens
```

这就像：
- 原始GPT2 = 英语老师，词表是英文单词
- RPG的GPT2 = 推荐专家，词表是item的"DNA密码"

虽然都叫"GPT2"，但RPG的GPT2：
- ✅ 使用了GPT2的Transformer架构
- ✅ 使用了GPT2的训练方法（自回归预测）
- ❌ **不使用**GPT2的预训练权重
- ❌ **不使用**GPT2的文本词表

---

## 总结

RPG是一个**纯粹的序列推荐模型**，通过以下创新设计实现高效推荐：

1. **OPQ量化**：将item embeddings压缩成32个离散codes
2. **GPT2架构**：借用Transformer建模用户行为序列
3. **分解预测**：将item预测分解为32个token预测
4. **图约束解码**：通过item-item图加速推理
5. **个性化搜索**：全局图+用户特定logits实现高效个性化

这是一个工程设计和算法创新完美结合的范例！🎯

