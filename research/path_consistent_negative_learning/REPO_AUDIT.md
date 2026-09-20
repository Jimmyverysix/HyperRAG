# WWW 2027 路径监督研究仓库审计

审计日期：2026-09-20

审计分支：`www-path-supervision-revision`
审计基线提交：`0640793`

## 1. 一句话研究论证

在遵循 HyperRAG 官方候选构造与负例采样流程的前提下，单条最短路径弱监督会把部分“未被选中但位于另一条完整主题—答案最短路径上的候选”作为强负例；本研究以唯一超参数 $\lambda$ 降低这类路径一致负例的监督强度，并通过匹配随机对照、数据集级验证选择、正式 Retriever/QA 评测和跨领域迁移检验覆盖率—排序精度折中。

证据边界是：路径一致性是结构属性，不自动等价于语义相关；结构化代理结果不能冒充正式 HyperRAG Retriever 或端到端 QA 结果；原始 $\lambda=0$ 实验的失败判定永久保留。

## 2. Git 与科研产物安全

- 上游官方仓库：`origin = https://github.com/Vincent-Lien/HyperRAG.git`。
- 协作仓库：`github = https://github.com/Jimmyverysix/HyperRAG.git`。
- 本轮从 `research/path-consistent-negative-learning` 的 `0640793` 创建独立分支 `www-path-supervision-revision`。
- 本轮开始前工作区已有一项未提交变化：`proposal/main.pdf` 从 Git 记录的 403,585 字节变为 403,592 字节，`main.tex` 未变化。该 PDF 被视为用户已有产物；本轮不还原、不覆盖，也不把它混入新投稿稿件提交。
- 下列历史目录与文件只读：`artifacts/audit/`、`artifacts/gate_c/`、`artifacts/weighted/`、`artifacts/provenance.json`、`proposal/main.tex` 及其历史 PDF。
- 新代码、运行记录和聚合结果统一进入独立命名空间；正式运行产物使用 `artifacts/www_revision/`，投稿稿件使用 `paper/www2027/`。

## 3. 仓库与依赖现状

官方实现位于以下目录：

- `HyperMemory/`：WikiTopics 超图构建与无训练检索。
- `HyperRetriever/`：WikiTopics 的数据准备、MLP Retriever 训练和端到端查询。
- `HyperRetriever_open/`：2WikiMultiHopQA、HotpotQA、MuSiQue 的开放域流程。
- `evaluate/`：WikiTopics 的 MRR/Hit@10，以及开放域 EM/F1。

根 `requirements.txt` 固定了 Python 依赖，但包含若干特定 Conda 构建路径，不能直接视为跨机器可复现环境锁文件。官方子目录的 requirements 更精简，但版本冻结不足。本轮需要增加研究专用环境说明，不改写官方 requirements。

未发现仓库内的独立 CCF-A skill；已读取并采用机器学习论文写作、Nature 式写作、科研绘图、学术检索和 PDF 检查规范。

## 4. 官方 HyperRAG 基线事实

### 4.1 数据准备

`HyperRetriever/retrieve/prepare.py`：

- 读取构建后的 `graph_chunk_entity_relation.graphml`。
- 对每个可达主题—答案实体对调用 `networkx.shortest_path` 选择一条最短路径。
- 正例是被选择路径上的候选转移。
- 负例从路径引导子图中随机采样，数量与正例匹配，并排除正例及其反向转移。
- 文本表示使用 `Alibaba-NLP/gte-large-en-v1.5`。
- 查询主题实体由 `gpt-4o-mini` 抽取并缓存。

### 4.2 Retriever 训练

`HyperRetriever/retrieve/train.py`：

- 模型为两层 MLP，输入为问题、头实体、超边、尾实体表示与距离编码。
- 官方损失为 `BCEWithLogitsLoss`。
- 默认训练超参数为 batch size 32、学习率 $10^{-4}$、最多 50 轮、patience 10。
- 官方代码以样本而不是问题为单位做 80/20 划分，只使用验证 BCE 做早停；正式修订必须改成问题级划分，防止同一问题的候选跨集合泄漏，同时保证 baseline 与加权方法共享相同划分。
- 官方代码自动使用可见 CUDA 设备，没有正式实验配置、逐运行 provenance 或多 GPU 任务队列。

### 4.3 官方评测

- WikiTopics：`evaluate/qa_eval_MRR_HIT.py` 报告 MRR 与 Hit@10。
- 开放域：`evaluate/qa_eval_EM_F1.py` 报告 EM、F1、precision、recall。
- 正式论文必须区分 Retriever 指标与生成后 QA 指标；当前结构代理的 answer reachability@10 和 selected-path PR-AUC 只能作为机制分析指标。

## 5. 数据与端到端资产

本地存在：

- `dataset/wikitopics_test_sampled/`：11 个领域的 1% 测试样本。
- `dataset/open_domain_dataset/`：2WikiMultiHopQA、HotpotQA、MuSiQue 语料与问题，共约 15.2 MB。
- `dataset/open_domain_splitted/`：三个开放域数据集的预划分问题，共约 0.4 MB。

本地缺失：

- 完整 `dataset/WikiTopicsQE_NLG/`。
- `expr/` 中构建后的 GraphML、节点表示、Retriever 数据集和检查点。
- `results/` 中官方端到端输出。
- 被 Git 忽略的 `config.json`，因此本地没有可确认的端到端 API 配置。

历史 provenance 指向服务器上的完整 WikiTopics_QE 整数图与原始结构代理运行目录。是否存在官方 NLG、GraphML、GTE 表示或端到端检查点仍需在服务器执行只读核验；在此之前不能声称已具备 official end-to-end 条件。

## 6. 历史产物的程序化核验

历史 JSON 可正常解析，关键事实为：

- 11 个 WikiTopics 领域、90,000 个三跳训练问题。
- 组合候选共 872,576,411,856 个，其中 42,926,111 个路径一致争议候选，占约 0.00492%。
- 五个种子实际采样 10,156,714 个负例，其中 586,117 个存在路径一致争议，占约 5.77%。
- 逐领域实际采样争议率范围约为 1.19% 到 11.67%。
- 原始 $\lambda=0$ 相对 baseline 的 answer reachability@10 提升明显，但 selected-path PR-AUC 下降 1.3187 个百分点；历史门槛 C 结论为“不扩展完整训练”。
- 事后加权修订在教育领域选择 $\lambda=0.1$，随后通过奖项领域独立确认。
- 历史跨领域 transfer 中，$\lambda=0.1$ 相对 baseline 的等权宏平均 answer reachability 提升约 4.81 个百分点，selected-path PR-AUC 变化约为 -0.22 个百分点。

这些数值只能从历史 artifacts 读取，不得手工改写，也不得把 $\lambda=0.1$ 重新定义为所有数据集的正式最优参数。

## 7. 当前研究代码的可复用部分

- `distances.py` 已实现完整成对距离条件 $d_I(s,v)+2+d_I(u,a)=d_I(s,a)$，并保留局部降距条件作为消融。
- `wikitopics.py` 已实现官方整数图的确定性重建、候选模拟和组合计数。
- `strategies.py` 已实现 baseline、完全屏蔽、全部标正、匹配随机屏蔽、路径一致加权和匹配随机加权。
- `structured_data.py` 已实现问题级划分、固定候选与逐题匹配随机对照。
- `statistics.py` 已实现以唯一问题为单位、先跨种子等权平均再配对自助法的统计。
- 历史选择、独立确认与跨领域 transfer 均有自动判定和机器可读汇总。

## 8. 必须修正的工程与科研缺口

1. 加权 BCE 仍定义在 `train_structured.py` 内，职责不独立；正式实现应移入专用模块，并严格按当前 batch 的 `sum(weights)` 归一化。
2. 现有正式选择规则是教育域上的事后修订协议，使用 reachability 与 PR-AUC 保护线；它必须保留为历史 transfer 证据，不能代替新的 dataset-specific validation selection。
3. 尚无统一的配置模型、`lambda_star.json` 选择器、逐实验 environment/config/command 记录和 `artifacts/www_revision/runs/<experiment_id>/` 结构。
4. 尚无论文所需的 per-query 争议分布、CSV 汇总、micro 补充统计、数据集级 $\lambda^*$ 与争议率相关性分析。
5. 尚无语义人工审计采样、盲化标注表和一致性分析脚本。
6. 当前 `proposal/main.tex` 是内部研究报告，正文仍以策略1/2/3和门槛 A–D 组织，并包含不适合投稿主线的论文—代码差异讨论；历史文档应保留，新投稿稿件需另建单线叙事。
7. 当前引用仅有 5 条，Related Work 不足；所有新增引文必须由 DOI、ACM、DBLP、arXiv 或出版社页面核验，不能凭记忆生成。
8. 当前图只输出 PDF/PNG，尚缺统一的 SVG、色盲安全配色、forest plot、dataset-specific lambda 图和完整图注统计信息。
9. 尚无 Makefile 或等价的一键入口，README 仍以历史实验报告为主。
10. official end-to-end 尚未完成，且可能受完整 NLG/GraphML、GTE 模型缓存和 API 配置缺失影响。

## 9. WWW 2027 约束

官方 Research Track 要求英文双栏 ACM 模板，推荐 `\documentclass[sigconf, anonymous, review]{acmart}`；主文 8 页，参考文献和可选附录后总页数不超过 12 页；第一页必须明确 Web relevance。目标 track 可定位为 “Search, Recommendation, and Retrieval-Augmented AI” 或 “Semantics and Knowledge”。正式截止日期为 2026-10-25 AoE。来源：[WWW 2027 Research Track Papers](https://www2027.thewebconf.org/research-track-papers/)。

当前阶段按用户要求先维护中文稿，但目录和篇幅应能直接映射到匿名英文 ACM 投稿版。

## 10. 当前阻塞项

- 服务器端官方 NLG、GraphML、GTE 表示和检查点状态尚未核验。
- 端到端生成需要可用且经用户授权的 API 配置；不得把结构代理结果改写为 QA 结果。
- 语义审计需要两名真实人工标注者；代码可以完成抽样和分析，但不能代替人工标签。
- 当前已有未提交 `proposal/main.pdf` 必须持续保留，直到用户决定如何处理。

## 11. 审计结论

仓库已具备可靠的路径一致性定义、历史结构审计、受控代理实验和统计基础，适合继续升级；但它还不是 WWW 可投稿工程。P0 可以立即开展，official end-to-end 能否完整执行取决于服务器资产和 API 配置核验。历史结论必须保持为：原始完全屏蔽失败，$\lambda=0.1$ 是后续固定权重 transfer 证据，而不是新的 dataset-specific 主结果。

## 12. 基线与服务器核验补记

- 2026-09-20 在本地运行历史测试：64 项全部通过，用时 3.023 秒。
- 服务器可见 8 张 RTX 3090；本研究继续严格只使用 GPU 0--5。检查时 0--5 均无显存占用。
- 服务器保留 1.4 GB 的完整 11 领域 `WikiTopics_QE` 整数图，以及历史 audit、Gate C、加权选择、确认与跨领域原始运行目录。
- 服务器仓库仍停在历史提交 `31db6f0`；新代码继续按约定经 GitHub 同步，不直接在服务器修改源码。
- 服务器同样缺少 `expr/`、`results/`、完整 `WikiTopicsQE_NLG`、GraphML、GTE 表示、官方 Retriever 检查点和 `config.json`。因此 official Retriever/QA 当前不是“尚未运行”，而是缺少上游官方资产和 API 配置，不能由现有代理实验替代。
