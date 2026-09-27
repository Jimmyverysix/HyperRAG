# HyperRAG 官方代码审计

审计日期：2026-09-27

官方仓库：`https://github.com/Vincent-Lien/HyperRAG`

冻结提交：`6d5a9033353c516a9220d78591f2c666f19ee0b1`

官方提交时间：2026-07-11 22:07:42 +0800

本地获取日期：2026-09-27

## 1. 审计结论

当前仓库中的 `HyperRetriever/`、`HyperRetriever_open/`、`HyperMemory/` 和 `evaluate/` 与上述官方提交在研究逻辑上相同；三个脚本仅有文件末尾换行差异，没有语义改动。因此，正式基线可以直接以该提交为上游锚点，后续研究修改应采用独立 wrapper 或最小补丁。

官方 WikiTopics 数据已完整取得到本地忽略目录：11 个领域、121 个 NLG 文件，总计约 1.35 GB；官方整数图也已完整取得。当前缺少的是构建后的 `expr/`、GTE 表示、Retriever 检查点和可用的 `gpt-4o-mini` API 配置，而不是 WikiTopics 原始数据。

## 2. WikiTopics 正例与负例构造

入口为 `HyperRetriever/retrieve/prepare.py`。

### 2.1 正例

1. 从 `train_queries.json` 和 `train_answers_hard.json` 读取问题及答案。
2. 通过 `gpt-4o-mini` 从问题中抽取主题实体，并只保留超图中存在的实体。
3. 对每个可达的主题实体—答案实体对调用一次 `networkx.shortest_path`。
4. 把返回路径上形如 `(头实体, 超边, 尾实体)` 的转移并入该问题的正例集合。

当存在多条等长最短路径时，官方代码只保留 NetworkX 按当前邻接插入顺序返回的一条路径，没有显式的 tie-breaking 参数。这正是 Path-selection Sensitivity 实验要测量的来源。

### 2.2 负例

1. 先从主题实体逐跳扩展；每一跳纳入当前实体关联的全部超边及这些超边关联的全部实体。
2. 只有位于已选最短路径上的实体会继续向下一跳扩展。
3. 在所得路径引导子图中随机选超边，再随机选两个不同的关联实体组成候选转移。
4. 排除正例及其反向转移，去重后采样至与正例数量相同，最多尝试 `20 × 正例数` 次。

因此，代码并不是把“每一跳头实体关联的所有其余转移”全部作为负例，而是从路径引导子图中随机采样等量负例。正式实验必须遵循实际代码；论文中若描述这一过程，应以代码行为为准。

固定 `random.seed(42)` 只固定 Python 随机采样。图节点和边的插入顺序仍会影响等长最短路径的选择，因此“采样种子复现”和“路径选择变体”必须分开控制。

## 3. 表示、Retriever 输入与模型

- 图节点、超边和问题均由 `Alibaba-NLP/gte-large-en-v1.5` 编码。
- `HyperRetriever/retrieve/model/emb.py` 取最后隐层的 `[CLS]` 表示并做 L2 归一化；默认维数为 1024。
- `DDEEncoder(max_hops=3)` 在候选子图上做正向与反向距离传播，给每个候选提供头尾实体的距离分布编码，共 30 维。
- 单个候选的输入为问题、头实体、超边、尾实体四个 1024 维向量与 30 维 DDE 的拼接，即 4126 维。
- 模型是两层 MLP：`4126 → 256 → 1`，中间为 ReLU。正式研究不需要、也不允许增加新网络模块。

`retrieve_dataset.py` 会把逐问题 JSON 展平为逐候选样本，只保留 `features` 与 `label`。问题标识、原始三元组、答案实体和路径一致标记均在此处丢失；后续最小补丁必须在不改变候选集合和特征的前提下保留这些元数据。

## 4. 训练、损失与划分

官方 `HyperRetriever/retrieve/train.py` 的设置为：

- `BCEWithLogitsLoss`，默认对 batch 求均值；
- Adam，学习率 `0.0001`；
- batch size 32；
- 最多 50 轮；
- patience 10，`min_delta=0.00001`；
- 按 validation BCE 早停。

训练脚本只读取训练问题生成的 `retrieval_dataset.pt`，随后按候选样本做 80%/20% 分层划分，`random_state=42`。同一问题的候选可能同时进入内部 train 和 validation。这是官方实现的真实行为，不应隐去；首次基线复现保持不变。正式方法比较也应共享同一内部划分，避免把划分变化误认为监督变化。

WikiTopics 发布数据另有 `valid_*` 与 `test_*`。官方训练脚本没有使用独立 valid 文件，也没有提供 validation 查询脚本。正式 λ 选择将通过 wrapper 在官方 valid 问题上运行与 test 相同的端到端 MRR/Hit 流程；test 在 λ 冻结前不运行。

## 5. 推理与评价

`HyperRetriever/wikitopics_query.py`：

1. 用 `gpt-4o-mini` 从问题中抽取查询实体/关键词；
2. 加载图表示、完整 GraphML、DDE 编码器和两层 MLP；
3. 对超图候选打分并按官方阈值与扩展逻辑构造上下文；
4. 再用 `gpt-4o-mini` 生成排名答案列表；
5. 输出问题、easy/hard 标准答案与预测列表。

`evaluate/qa_eval_MRR_HIT.py` 对生成后的排名答案列表计算百分制 MRR 与 Hit@10。它评估的是端到端 QA 答案排名，不是 MLP 候选分数本身。当前结构代理的 answer reach@10 与 PR-AUC 不能替代这两个官方指标。

## 6. 检查点

训练保存的 `best_retrieval_model.pth` 是包含以下字段的字典：

- `model_state_dict`
- `pred_in_size`
- `emb_size`

`HyperRetriever/hypergraphrag/operate.py` 的按需加载路径正确读取上述字段；但 `HyperRetriever/hypergraphrag/hypergraphrag.py` 的预加载路径把整个字典直接传给 `load_state_dict`。只要检查点存在，官方 WikiTopics 查询构造器就会在预加载阶段失败。基线复现需要一个只修复检查点解包的最小补丁，不能改动模型参数、候选、打分或评价。

## 7. 外部依赖

官方全流程依赖两类外部资产：

- Hugging Face 上的 `Alibaba-NLP/gte-large-en-v1.5`，用于图节点、超边和问题表示；
- OpenAI `gpt-4o-mini`，用于超图构建时的信息抽取、训练数据的主题实体抽取，以及推理时的查询理解与答案生成。

本地和服务器均未发现 `OPENAI_API_KEY` 或 `config.json`。服务器的 `sdhp` 环境已有 CUDA PyTorch、NetworkX 与 scikit-learn，但缺少 `transformers`、`requests` 和 `openai`。服务器可连接 GitHub，但当前对 Hugging Face、OpenAI 与 Google Drive 的直接访问失败；正式运行需复用本地网络代理或预先同步模型缓存。网络可达性与 API 授权是两个独立条件。

## 8. 开放域流程审计

`HyperRetriever_open/` 支持 `2wikimultihopqa`、`hotpotqa`、`musique` 的训练、查询与 EM/F1 评价，但从全新克隆无法原样跑通：

1. 仓库没有把 `*_corpus_sentences.jsonl` 构建成开放域 GraphML 的入口脚本。
2. `retrieve/prepare.py` 读取不存在的 `dataset/open_domain_dataset/open_domain_splitted_query`，实际发布路径为 `dataset/open_domain_splitted`。
3. 发布数据中的 `answer` 是字符串，prepare 脚本却逐元素迭代，实际会按字符查找答案实体。
4. `operate.py` 的 GraphML 预加载路径是 `expr/{domain}`，按需回退路径却是 `expr/open_domain/{domain}`。
5. 发布数据只有 800 个 train 与 200 个 test 问题，没有独立 validation 文件。

三个数据集的代码完整度相同。正式外部 benchmark 冻结为 `2wikimultihopqa`：其发布语料最小，能够减少首次完整复现的构图成本，且官方论文提供了对应的 HyperRetriever EM/F1 参照。需要的修复仅限路径、字符串答案处理、GraphML 构建入口和从 train 中确定性划出的 validation；不改变模型、候选、推理或评价定义。

## 9. 基线复现状态

截至 2026-09-27，官方基线尚未成功复现，状态不是“结果为零”，而是“尚未产生结果”：

- 完整 WikiTopics 数据：已具备；
- 官方源代码与上游锚点：已具备；
- 服务器 0–5 号 RTX 3090：可用；
- 构建后的官方超图与检查点：缺失；
- `gpt-4o-mini` API 配置：缺失；
- 服务器 Python 依赖与 GTE 模型缓存：尚未安装/同步；
- 服务器对模型与 API 站点的直接网络：不可用，需代理。

在 API 配置到位并完成一个领域的官方 baseline 训练、推理、评价之前，不修改官方监督逻辑，也不启动 11 领域 × 6 λ × 5 seeds 的正式任务。
