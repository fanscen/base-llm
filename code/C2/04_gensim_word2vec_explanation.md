# Word2Vec 模型实战代码解释

> 本文解释 `04_gensim.ipynb` 中 "五、Word2Vec模型实战" 章节的代码

---

## 5.1 模型训练与核心参数

### Step 1: 准备语料

```python
headlines = [
    # 财经 (8条)
    "央行降息，刺激股市反弹",
    "A股市场持续震荡，投资者需谨慎",
    ...
    # 体育 (8条)
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    ...
]

tokenized_headlines = [list(jieba.cut(title)) for title in headlines]
```

**作用**：准备包含两个主题领域的语料，用于训练词向量

**语料规模**：
- 16 条新闻标题
- 财经类 8 条 + 体育类 8 条

---

### Step 2: 训练 Word2Vec 模型

```python
model = Word2Vec(
    tokenized_headlines,  # 分词后的语料
    vector_size=50,       # 词向量维度
    window=3,             # 上下文窗口大小
    min_count=1,          # 最小词频阈值
    sg=1                  # 训练算法：1=Skip-gram, 0=CBOW
)
```

#### 核心参数详解

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|----------|
| `vector_size` | 100 | 词向量维度 | 小语料: 50-100，大语料: 200-300 |
| `window` | 5 | 上下文窗口大小 | 短文本: 3-5，长文本: 5-10 |
| `min_count` | 5 | 忽略词频 < min_count 的词 | 小语料设 1，大语料设 5-10 |
| `sg` | 0 | 训练算法 | 0=CBOW(快), 1=Skip-gram(准) |
| `workers` | 3 | 并行线程数 | CPU 核心数 |
| `epochs` | 5 | 训练轮数 | 5-15 |

---

### 两种训练算法对比

#### CBOW (Continuous Bag of Words)

```
上下文词 → 预测 → 中心词
```

- **原理**：用周围词预测中心词
- **特点**：训练快，适合大语料
- **公式**：$P(w_t | w_{t-c}, ..., w_{t+c})$

#### Skip-gram

```
中心词 → 预测 → 上下文词
```

- **原理**：用中心词预测周围词
- **特点**：对小语料和稀有词效果更好
- **公式**：$P(w_{t-c}, ..., w_{t+c} | w_t)$

**代码中使用 `sg=1`，即 Skip-gram**，因为语料较小。

---

### 训练结果

```python
print(f"词汇表大小: {len(model.wv)}")
print(f"词向量维度: {model.wv.vector_size}")
```

**输出示例**：
```
模型训练完成！
词汇表大小: 85
词向量维度: 50
```

**说明**：
- `model.wv` 是 `KeyedVectors` 实例，存储所有词向量
- 每个词被映射为 50 维的稠密向量

---

## 5.2 使用词向量

### 功能 1: 寻找最相似的词

```python
similar_to_market = model.wv.most_similar('股市')
print(f"与 '股市' 最相似的词: {similar_to_market}")
```

**输出示例**：
```
与 '股市' 最相似的词: 
[('市场', 0.89), ('投资者', 0.85), ('震荡', 0.82), ('反弹', 0.78), ...]
```

**原理**：计算余弦相似度

$$\text{similarity}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \times |\vec{b}|}$$

---

### 功能 2: 计算两个词的相似度

```python
similarity = model.wv.similarity('球队', '球员')
print(f"'球队' 和 '球员' 的相似度: {similarity:.4f}")
```

**输出示例**：
```
'球队' 和 '球员' 的相似度: 0.8523
```

**应用**：同义词检测、语义相关性判断

---

### 功能 3: 获取词向量

```python
market_vector = model.wv['市场']
print(f"'市场' 的向量维度: {market_vector.shape}")
```

**输出示例**：
```
'市场' 的向量维度: (50,)
```

**应用**：作为下游任务的输入特征

---

### 其他常用操作

```python
# 找出不合群的词
model.wv.doesnt_match(['股市', '基金', '球员', '债券'])
# 输出: '球员'（其他都是财经词）

# 类比推理: 国王 - 男人 + 女人 = 王后
model.wv.most_similar(positive=['国王', '女人'], negative=['男人'])

# 检查词是否在词汇表中
'股市' in model.wv  # True
```

---

## 5.3 模型的持久化

### 保存词向量

```python
model.wv.save("news_vectors.kv")
```

**说明**：
- 只保存词向量（`KeyedVectors`），不保存训练器
- 文件格式：`.kv` (KeyedVectors 二进制格式)
- 文件更小，加载更快

---

### 加载词向量

```python
loaded_wv = KeyedVectors.load("news_vectors.kv")
```

**使用加载的向量**：

```python
loaded_wv.similarity('球队', '球员')  # 与原模型结果一致
```

---

### 其他保存方式

```python
# 保存完整模型（包含训练器，可继续训练）
model.save("word2vec.model")

# 加载完整模型
from gensim.models import Word2Vec
loaded_model = Word2Vec.load("word2vec.model")

# 导出为文本格式（人类可读）
model.wv.save_word2vec_format("vectors.txt", binary=False)

# 导出为二进制格式（更紧凑）
model.wv.save_word2vec_format("vectors.bin", binary=True)
```

---

## Word2Vec 核心原理

### 分布式假设

> **"一个词的含义由它周围的词决定"**

```
"央行 降息 刺激 股市 反弹"
       ↑
    上下文决定"股市"的含义
```

### 训练目标

**Skip-gram 的目标函数**：

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} | w_t)$$

其中：

$$P(w_o | w_c) = \frac{\exp(\vec{v}_o \cdot \vec{v}_c)}{\sum_{w \in V} \exp(\vec{v}_w \cdot \vec{v}_c)}$$

**优化技巧**：
- **负采样 (Negative Sampling)**：将 softmax 转化为二分类问题
- **层次 Softmax (Hierarchical Softmax)**：用哈夫曼树加速

---

## Word2Vec vs TF-IDF vs LDA 对比

| 特性 | TF-IDF | LDA | Word2Vec |
|------|--------|-----|----------|
| **表示** | 稀疏向量 | 主题分布 | 稠密向量 |
| **维度** | 词汇表大小 | 主题数 | 自定义（50-300）|
| **语义** | 无 | 主题级 | 词级 |
| **相似度** | 不支持 | 主题相似 | 语义相似 |
| **训练方式** | 统计 | 生成模型 | 神经网络 |

---

## 应用场景

| 场景 | 使用方式 |
|------|----------|
| **相似词推荐** | `most_similar()` |
| **文本分类** | 词向量平均作为文档向量 |
| **命名实体识别** | 作为特征输入 CRF/BERT |
| **推荐系统** | 用户/物品向量相似度 |
| **机器翻译** | 跨语言词向量对齐 |