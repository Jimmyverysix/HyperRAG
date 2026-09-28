# Retriever-only 零 LLM 研究仓库审计

审计日期：2026-09-28

审计分支：`codex/www-retriever-only`

审计起点：`2cb708d`

## 1. 审计结论

本轮正式研究不再等待生成式模型 API，也不复现 HyperRAG 的答案生成阶段。研究对象被收窄为：在确定性重建的 WikiTopics 超图上，复用官方 HyperRetriever 的 GTE 文本表示、DDE 结构编码和两层 MLP，检验单路径弱监督是否对路径一致负例施加了过强的负监督。

正式实验只改变路径一致负例的损失权重。答案实体仅用于构造训练监督和计算 Retriever-only 指标，不进入候选生成、特征构造或候选打分。历史结构代理 artifacts、历史 proposal 和用户已有的 `proposal/main.pdf` 均保持只读。

## 2. Git 与历史 provenance

- 协作远端为 `https://github.com/Jimmyverysix/HyperRAG.git`，官方远端为 `https://github.com/Vincent-Lien/HyperRAG.git`。
- 官方上游固定为提交 `6d5a9033353c516a9220d78591f2c666f19ee0b1`。
- 本轮从 `2cb708d` 新建 `codex/www-retriever-only`，与上一轮依赖生成式 API 的 `codex/www27-official-hyperretriever` 分离。
- 开始时唯一未提交文件是用户已有的 `proposal/main.pdf`；本轮不覆盖、不暂存、不提交该文件。
- `artifacts/audit/`、`artifacts/gate_c/`、`artifacts/weighted/`、`artifacts/www_revision/` 与 `proposal/` 是历史记录，不作为新实验写入位置。
- 新 raw runs、聚合结果、表格和图片统一使用 `artifacts/retriever_only/` 命名空间。

## 3. 历史结果复核

程序化历史核验共检查 16 项一致性条件，全部通过。现有 85 项单元测试全部通过。可继续使用但不得越界解释的历史事实包括：

- 11 个领域共约 90,000 个三跳训练问题；
- 实际采样负例中约 5.77% 是路径一致负例；
- 约 63.77% 的问题在五个采样种子中至少一次受到影响；
- 结构代理中完全屏蔽提高 answer reach@10，但旧门槛实验出现 selected-path PR-AUC 下降；
- 历史数据集级选择中 10/11 个领域选择 `lambda=0.00`。

这些结果只定位为结构诊断证据，不能充当本轮 GTE + DDE + MLP 的正式语义 Retriever 结果。

## 4. 数据与本地模型资产

本地忽略目录和服务器均具备完整数据：

- WikiTopics_QE 整数图：11 个领域；
- WikiTopics NLG：11 个领域、121 个文件；
- `Alibaba-NLP/gte-large-en-v1.5` 模型权重与远程代码缓存；
- 服务器独立环境 `hyperrag_official`，能够离线加载 GTE 并识别 CUDA。

服务器提供 8 张 RTX 3090。本研究仍只允许使用 GPU 0--5，每个任务独占一张 GPU，不使用 DDP。审计时 GPU 1--5 空闲，GPU 0 有其他进程占用；调度器不得抢占已有进程。

## 5. 官方 Retriever 可复用边界

`HyperRetriever/retrieve/model/emb.py` 使用 GTE 最后一层的 `[CLS]` 向量并做 L2 归一化；`model/dde.py` 生成 30 维 DDE；`model/mlp.py` 是 `4126 -> 256 -> 1` 的两层 MLP。正式实现复用同一模型与表示定义，并为设备、缓存路径和批处理增加研究侧入口，不修改网络结构。

官方训练设置为 Adam、学习率 `0.0001`、batch size 32、最多 50 轮、patience 10、`min_delta=0.00001`。官方 prepare 对每个可达 topic-answer pair 选择一条最短路径，并从 path-guided subgraph 采样与正例等量的负例。正式 baseline 保留这些设置。

官方端到端推理还包含生成式实体抽取、图构建和答案生成。它们不属于本轮实验，也不得通过替代 API 重新引入。

## 6. 确定性图重建

正式图使用公开整数文件确定性重建：

1. 用 `og_mappings.pkl` 把整数实体和关系 ID 还原为 Wikidata ID；
2. 合并 `train_graph.txt` 与 `test_inference.txt`，与官方构图同时插入 train/test 文本的 transductive 范围一致；
3. 按 head 实体聚合外向事实为一个超边；
4. 实体文本使用冻结的英文标签；超边文本把同一 head 下的结构化事实按稳定顺序序列化为 `head | relation: tail; ...`，不重新生成自然语言；
5. topic entity 直接读取结构化查询 `(e,(r1,r2,r3))` 中的 `e`；
6. 自然语言查询直接使用现有 NLG 文件。

该实现是确定性 WikiTopics 超图上的官方 Retriever 组件实验，不声称重现原生成式 GraphML。

## 7. NLG 与结构查询对齐风险

NLG 转换脚本保留 pickle 中结构查询的迭代顺序，但会过滤缺少英文标签的 topic 或 relation。以 art/train 为例，结构文件有 10,000 个三跳查询，NLG 文件有 9,971 个问题；未经校验地按行号配对会在首个过滤位置后系统性错位。

修复限定为一次性数据对齐：从 Wikidata 官方服务获取非生成式英文标签快照，利用原转换脚本“保持顺序、只删除缺失标签查询”的性质，把结构化查询与现有 NLG 查询做单调对齐，并保存冻结对齐元数据。原始 graph sentences 没有保存 head 标识，且其生成时标签可用性与当前快照不同，不能可靠逐行回溯；因此正式图文本改用结构化事实的确定性序列化。正式特征准备只读取冻结快照，不访问网络。若查询无法完整对齐，数据准备立即失败，不启动 GPU 训练。

## 8. Retriever-only 候选与评价

训练候选严格来自官方 released prepare 逻辑。验证和测试候选不使用答案。真实规模探测表明，按当前 n-ary 超边直接三跳全展开时，art 的典型查询约产生 1,560 万个候选，无法作为 11 领域评价接口。因此评价固定使用答案不可见的 GTE 语义束：每跳按冻结 GTE 的 query--head--hyperedge--tail 平均余弦相似度保留 10 个转移，最多三跳；候选束对所有方法、lambda 和种子固定。正式 Retriever 仍直接输出这些候选的原始 MLP logit，不设置 logit 阈值。

主指标为 Answer-Path MRR：按 logit 排序逐一加入有向转移，记录 top-k 子图首次连接任一 topic entity 与任一正确 answer entity 的最小 `k`。次指标为 Answer Reach@10，并补充 Answer Reach@5。没有任何可达正确答案的查询记为 RR=0；因 topic 或全部答案不在重建图中而不具备评价条件的查询单独计数并排除，不能静默丢弃。

## 9. 需要实现的最小改动

1. 独立的 NLG/结构对齐与标签快照工具；
2. 确定性联合超图和三跳候选生成；
3. 复用官方 GTE、DDE、MLP 的特征缓存；
4. 保留 query key、triplet、path-consistent 标记和逐样本权重的数据集；
5. 按权重和归一化的 BCE；
6. retrieve-only scorer 与 Answer-Path 指标；
7. validation-only lambda 选择、逐题 matched random、三种路径择一变体；
8. query-level paired bootstrap、结果聚合、LaTeX 宏表和矢量图自动生成。

不增加 GNN、teacher、risk estimator、curriculum、第二个可学习参数或复合选择指标。

## 10. 当前状态

仓库、历史产物、数据与模型资产均满足启动零 LLM Retriever-only 研究的条件。上一轮的 API 阻塞已经失效。正式结果尚未产生；在新协议、数据对齐测试、路径一致测试、损失端点测试和 retrieve-only 指标测试通过前，不启动 11 领域 sweep。
