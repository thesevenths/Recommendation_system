Traditional recommendation algorithms, such as collaborative filtering, recall, coarse-ranking, and fine-ranking, have been developed over decades, and their technological  benefits/advantages have largely peaked. There is a growing need to explore new recommendation algorithms, such as leveraging Large Language Models (LLMs) or reinforcement learning to enhance the performance of recommendation systems.

* [SHAP：以淘宝展示广告点击率预估做可视化分析解释](https://www.cnblogs.com/theseventhson/p/18922913 "发布于 2025-06-13 10:17")
* [SHAP：以bitcoin price预测为例的机器学习算法可视化解析](https://www.cnblogs.com/theseventhson/p/18898661)
* [抖音推荐算法部分模型概述](https://www.cnblogs.com/theseventhson/p/18831881)
* [LLM大模型：推荐系统应用-HLLM实战&amp;DSIN](https://www.cnblogs.com/theseventhson/p/18820121)
* [LLM大模型: RAG两大核心利器: M3E-embedding和bge-rerank](https://www.cnblogs.com/theseventhson/p/18273943)



在推荐系统中，整个流程通常分为几个关键阶段：**召回（Recall）→ 粗排（Pre-ranking）→ 精排（Ranking）→ 重排（Re-ranking）**。每个阶段有不同的目标和算法选择，下面详细说明各阶段常见的算法及其优缺点。

---

### 一、召回（Recall）

**目标**：从海量物品库中快速筛选出几百到几千个与用户可能相关的候选物品，强调速度和覆盖率。

#### 常见算法：

1. **协同过滤（Collaborative Filtering, CF）**

   - 用户-物品协同过滤（User-based / Item-based）
   - 矩阵分解（如 SVD、SVD++）
   - 实现方式：基于用户行为计算相似度
2. **基于内容的召回（Content-Based）**

   - 利用物品的文本、标签、类别等特征进行匹配
   - 使用 TF-IDF、Word2Vec、BERT 等表示物品
3. **向量召回（Embedding-based）**

   - 双塔模型（Dual-Tower DNN）：用户塔和物品塔分别生成向量，通过近似最近邻（ANN）检索
   - 向量数据库：Faiss、Annoy、HNSW 等支持高效检索
4. **多路召回（Multi-Candidate Recall）**

   - 并行使用多种召回策略（如协同过滤、热门、地理位置、实时行为等），然后合并结果
5. **图神经网络（GNN）/ 图嵌入（Graph Embedding）**

   - DeepWalk、Node2Vec、PinSage 等，将用户-物品交互构建成图进行嵌入
6. **倒排索引 + 标签召回**

   - 基于用户兴趣标签或物品属性进行关键词匹配

---

#### 优点与缺点：

| 算法     | 优点                         | 缺点                                       |
| -------- | ---------------------------- | ------------------------------------------ |
| 协同过滤 | 简单有效，适合冷启动后的用户 | 数据稀疏时效果差，难处理新物品（冷启动）   |
| 内容召回 | 解决新物品冷启动问题         | 忽视用户偏好多样性，可能陷入信息茧房       |
| 向量召回 | 支持语义匹配，可扩展性强     | 训练复杂，需要高质量 embedding 和 ANN 工具 |
| 多路召回 | 提高覆盖率和多样性           | 需要调权衡不同策略的融合策略               |
| 图方法   | 捕捉高阶关系，结构信息丰富   | 计算开销大，训练复杂                       |

---

### 二、粗排（Pre-ranking）

**目标**：对召回的几百~几千个物品进行初步打分排序，减少到百以内，为精排做准备。要求比精排快，但比召回更精细。

#### 常见算法：

1. **简化版深度模型**

   - 蒸馏模型（Knowledge Distillation）：用精排模型作为老师，训练轻量学生模型
   - 简化结构的 DNN 或 Wide & Deep 的轻量版本
2. **Factorization Machines (FM) / DeepFM 轻量版**

   - 能建模特征交叉，适合点击率预估
3. **LR + 特征交叉**

   - 传统但高效，常用于资源受限场景
4. **向量内积打分（双塔打分）**

   - 用户向量和物品向量直接点积，无需 ANN 检索

---

#### 优点与缺点：

| 算法        | 优点                   | 缺点                         |
| ----------- | ---------------------- | ---------------------------- |
| 蒸馏模型    | 性能接近精排，速度快   | 依赖精排模型质量，训练复杂   |
| FM / DeepFM | 支持特征交叉，效果较好 | 比 LR 复杂，需工程优化       |
| LR          | 极快，可解释性强       | 表达能力弱，依赖人工特征工程 |
| 向量点积    | 推理极快，适合线上服务 | 打分粒度较粗，缺乏上下文特征 |

---

### 三、精排（Ranking）

**目标**：对粗排后的候选集（通常 < 100）进行精准打分，预测 CTR、CVR、停留时长等目标。

#### 常见算法：

1. **逻辑回归（LR） + GBDT**

   - GBDT 自动生成特征组合，LR 做最终预测（如 Facebook GBDT+LR）
2. **Wide & Deep**

   - 结合记忆（wide部分）和泛化（deep部分）
3. **DeepFM / xDeepFM**

   - 替代人工特征交叉，自动学习高阶特征交互
4. **DIN / DIEN / BST / SIM**

   - 引入注意力机制，捕捉用户历史行为中的兴趣变化
   - DIN：关注相关商品；DIEN：用 GRU 建模兴趣演化
5. **MMoE / PLE**

   - 多任务学习，同时优化 CTR、CVR、点赞、收藏等多个目标
6. **Transformer-based 模型（如 AliExpress 的 MIMN、TDM、SIM）**

   - 处理长序列用户行为，建模长期兴趣
7. **图神经网络（Graph-based Ranking）**

   - 利用用户-物品图结构增强表示（如 Uber Graph Rec）

---

#### 优点与缺点：

| 算法        | 优点                         | 缺点                         |
| ----------- | ---------------------------- | ---------------------------- |
| LR + GBDT   | 成熟稳定，易于部署           | 特征工程依赖强，表达能力有限 |
| DeepFM      | 自动特征交叉，性能好         | 对稀疏特征敏感，易过拟合     |
| DIN/DIEN    | 能捕捉动态兴趣               | 模型复杂，训练成本高         |
| MMoE/PLE    | 多任务共享知识，提升整体效果 | 模型设计复杂，需平衡任务权重 |
| Transformer | 建模长序列能力强             | 显存消耗大，推理慢           |
| GNN         | 捕捉复杂关系                 | 训练难，图构建成本高         |

---

### 四、重排（Re-ranking）

**目标**：在精排基础上调整顺序，考虑多样性、新颖性、业务规则、公平性等。

#### 常见算法与策略：

1. **多样性重排**

   - MMR（Maximal Marginal Relevance）：平衡相关性和多样性
   - 基于类别的打散（如每类最多展示2个）
2. **强化学习（RL）**

   - 如 Google 的 Multi-Armed Bandit、DRR（Diversity Reward Reinforcement）
   - 学习长期用户满意度
3. **序列建模重排**

   - Rerank with Transformer / GRU：建模展示序列的整体体验
4. **规则引擎**

   - 插入广告、运营位、去重、打散热门等
5. **Learning to Rank（LTR）**

   - Listwise 方法：Softmax Loss、ListNet、LambdaMART
   - 考虑列表整体排序质量
6. **Fairness / Debiasing 重排**

   - 抑制马太效应，提升小众物品曝光机会

---

#### 优点与缺点：

| 方法     | 优点                      | 缺点                   |
| -------- | ------------------------- | ---------------------- |
| MMR      | 提升多样性，避免重复      | 可能牺牲相关性         |
| RL       | 优化长期收益              | 训练不稳定，样本效率低 |
| LTR      | 优化排序指标（NDCG、MAP） | 需要标注数据或模拟反馈 |
| 规则引擎 | 可控性强，符合业务需求    | 灵活性差，维护成本高   |
| 序列建模 | 考虑上下文影响            | 推理延迟增加           |

---

### 总结对比表：

| 阶段 | 目标         | 典型算法                       | 关键考量             |
| ---- | ------------ | ------------------------------ | -------------------- |
| 召回 | 海量中选几百 | 协同过滤、向量召回、多路召回   | 速度、覆盖率、冷启动 |
| 粗排 | 初步打分筛选 | 蒸馏模型、FM、双塔点积         | 效率 vs 准确性       |
| 精排 | 精准打分排序 | DIN、DeepFM、MMoE、Transformer | 模型表达能力、多任务 |
| 重排 | 优化最终展示 | MMR、RL、LTR、规则             | 多样性、公平性、体验 |

---

### 小tips：

- **工业级系统通常是混合架构**：多路召回 → 蒸馏粗排 → 多任务精排 → 规则+学习重排。
- **冷启动问题**：召回阶段可用内容/热度/社交关系补足。
- **实时性**：引入实时行为（如 session-based 召回）提升响应速度。
- **可解释性与可控性**：重排阶段更适合加入业务规则干预。
