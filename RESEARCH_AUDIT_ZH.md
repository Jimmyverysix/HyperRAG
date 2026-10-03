# 路径一致负例研究审计

日期：2026-10-03。工作分支：`codex/pcn-cross-dataset`，基于 `99cd1c4`。本轮开始时工作树已有未提交的论文、脚本和产物改动；保留原状，只提交本轮明确新增或修改的文件。旧论文与所有已有实验目录不覆盖。

## 1. 项目结构与研究边界

原项目包含 `HyperMemory*`（建图与问答）、`HyperRetriever*`（检索器）、`evaluate/` 和 `dataset/`。当前研究主线位于 `research/path_consistent_negative_learning/`，是冻结 GTE、官方 DDE、两层 MLP 的 Retriever-only 受控实验；APC-MRR 衡量完整答案路径的完成排名，不是实体排序 MRR，也不是端到端问答正确率。

`structured_data.py`、`train_structured.py`、`artifacts/gate_c`、`artifacts/weighted` 为早期结构代理实验。它们具有真实历史产物，但不混入当前 GTE+DDE+MLP 主结果。

## 2. 实际代码路径

| 环节 | 当前实现 | 审计结论 |
|---|---|---|
| 原始训练入口 | `HyperRetriever/retrieve/prepare.py`、`train.py` | 上游单路径训练参考；本轮不直接改写 |
| 正式编排 | `scripts/retriever_only_pipeline.py`、`build_retriever_manifest.py`、`run_retriever_queue.py` | 可执行准备、训练、评价队列 |
| WikiTopics 数据与对齐 | `retriever_only/data.py`、`alignment.py`、`labels.py` | 当前对齐只使用主题标签与发布顺序；英文标签用于编码，QID 用于实体身份 |
| 超图构造 | `retriever_only/graph.py` | 按 head 聚合事实；实体—事实关联图；一次逻辑转移成本为 2；固定传导式图包含发布的训练图及测试推理上下文 |
| 训练候选 | `retriever_only/candidates.py` | 每个答案选一条确定最短路径；正转移取并集；路径引导子图产生负候选池 |
| 实际负采样 | `sample_negative_transitions` | 等量采样；排除所选正转移及其反向；有限尝试可能少于目标数，统计应使用实际数量 |
| PCN | `shortest_path_dag_nodes`、`path_consistent_transitions_from_dag` | 检查源距离增加 2 且终点属于完整答案最短路径 DAG；只对实际 sampled negatives 定义 PCN；不依赖局部答案降距 |
| 共享张量 | `retriever_only/prepared.py` | `PreparedCandidates` 保存固定候选、原标签、PCN mask 和特征索引 |
| 损失策略 | `method_weights`、`path_supervision/weighted_loss.py` | Baseline 全 1；Ours 将 PCN 权重设 lambda；Random 逐题从实际负例中均匀抽取恰好同数量；lambda=0 为屏蔽 |
| 训练 | `retriever_only/training.py` | 官方 4126→256→1 MLP，Adam；候选级 80/20 分层内部留出早停；全屏蔽 batch 跳过优化器更新 |
| 推理候选 | `semantic_beam_candidates` | 只读 topic 与冻结 GTE 相似度；固定 beam=32，最多三跳；不用答案或训练后分数 |
| 指标 | `retriever_only/metrics.py`、`evaluation.py` | 有向候选转移按分数排序；逐步插入边直到到达答案；保留 Candidate Oracle |
| 统计 | `retriever_only/aggregation.py`、`statistics.py` | 同题先平均五种子，再做 10,000 次配对 bootstrap；WikiTopics 按领域等权 |

普通图对应单位转移成本为 1，完整条件为 `d(s,v)+1+d(u,a)=d(s,a)`；超图条件为 `d_I(s,v)+2+d_I(u,a)=d_I(s,a)`。两者都表示结构成员资格，不自动给出语义正标签。

## 3. 两个现有基准的支持程度

**WikiTopics：正式支持。** 冻结配置在 `configs/retriever_only/wiki_main.json`；11 个领域、共享种子 42–46。完整原始数据、编码张量、prepared candidates 与 checkpoint 在服务器 `/root/hyperrag_pcneg/`。正式修正版运行根为 `runs/final_revision_answer_free`，本地聚合在 `artifacts/final_revision/`。固定屏蔽主结果来自 `fixed_masking/final_main_results.json`，不从论文转录。

**MetaQA-3hop-vanilla：已有真实完整实验。** `artifacts/final_confirmation/external/` 有普通有向图 adapter、准备脚本、五种子三策略训练报告、逐题评价 CSV、JSON 和聚合。服务器根为 `runs/final_confirmation_metaqa`，数据源为 `data_sources/MetaQA_3hop`。第 42 种子的准备报告记录 114,196 个训练问题、2,832,418 个实际采样负例、334,303 个 PCN；test 为 14,274 题。`external_summary.json` 的 APC-MRR 分别为 Baseline 20.8621%、Random 20.6575%、Mask 22.1851%。这是本项目监督机制在官方 KG/split 上的受控迁移，不是 MetaQA 官方模型复现。

旧 `NEW_CHAT_MEMORY.md` 声称第二基准未完成，已经落后于现有 MetaQA 产物。本轮以原始代码、运行记录和结果文件为准，不继承旧 PDF 或旧交接文档中的废弃历史叙事。

## 4. 现有基线与真实结果位置

正式 GTE+DDE+MLP 主结果目前只有 Original、Matched Random、PCN Mask；早期结构代理虽然存在 positive 策略，但不能替代正式 Positive Relabel 实验。

- WikiTopics 固定屏蔽：`artifacts/final_revision/fixed_masking/final_main_results.json`、`final_main_results.csv`、`fixed_masking_results.csv`。
- WikiTopics 候选上限与验证权重：`artifacts/final_revision/candidate_oracle/`、`lambda_validation/`。
- MetaQA：`artifacts/final_confirmation/external/reports/`、`prepared/`、`eval/`、`external_results.csv`、`external_summary.json`。
- 可达子集已有分析：`artifacts/final_confirmation/chinese_polish/oracle_reachable_results.csv`。
- 历史结果：`artifacts/audit`、`gate_c`、`weighted`、`www_revision`、`retriever_only`；保留来源标签。

现有 WikiTopics 汇总报告固定屏蔽 APC-MRR 约 6.312%、Reach@10 约 12.858%；这里只作为审计定位，新增论文表格必须由结果脚本生成。尚无两个新增基准的真实主结果，尚无正式四策略跨基准表，不能将这些缺口写成已完成。

## 5. 当前论文

可读中文稿为 `paper/www2027/main_readable.tex`，其旧结构为引言、屏蔽方法、实验与补充分析；相关工作有注释块，需要重建完整论证。BibTeX 为同目录 `references.bib`；宏与表在 `generated/`。此前 `main.tex/main.pdf` 在工作树被删除，这属于已有现场，不自行恢复或覆盖。新稿另建 `paper_restructured_zh.tex`，旧稿保留。

## 6. 已发现的实际技术债务

1. **正式训练缺 Positive Relabel。** `method_weights` 只支持三策略，不能改变训练标签；需要共享的 `(labels, weights)` 策略入口，并保持内部划分使用原标签，避免不同策略改变训练/早停样本。
2. **MetaQA 训练循环重复。** adapter 和 runner 位于 artifacts，另复制逐 batch 训练、评价与聚合逻辑；应将可复用数据接入移入源码，接入同一策略与评价框架，历史副本保持只读。
3. **WikiTopics 评价不是所有原始问题。** `prepare_evaluation_candidates` 会跳过不支持的对齐、图外 topic、以及全部答案不在图中的问题。现有成绩的分母是 eligible queries；新稿必须披露，不得称为所有官方问题。新增 adapter 应对有 topic 的官方评价问题保留图外/不可达答案并记零，答案不得决定候选是否生成。
4. **发生率口径需统一。** WikiTopics 某些受影响比例为五种子至少一次出现，MetaQA 现有摘要主要记录每种子 PCN 总数。跨基准表需要同时记录唯一训练问题数、种子累计实际负例、种子累计 PCN、逐种子和至少一次受影响题，避免混用分母。
5. **独立新图尚未接入。** Freebase 相关基准不能只下载 QA 文件就宣称适配完成，需要真实可用的固定图与可计算路径；按 gold query 剪枝的子图不能冒充 answer-free 检索。
6. **小 MLP 与大特征内存制约吞吐。** WikiTopics 已一次性物化特征；MetaQA 5.66M 候选的完整 4126 维 FP32 特征约 93.5 GB，不能在单卡上物化。需要以共享固定表示、批次预取/分块及多任务调度提速，不能为提速偷偷改变策略间 batch、精度或候选。

## 7. 服务器与执行计划

已通过用户授权 SSH 读取资源。服务器有 8 张 RTX 3090、64 个 CPU 逻辑线程；首次探测 GPU 1 有已有进程，其余空闲。最多选择六张可用卡，计划优先使用 0、2、3、4、5、6；不终止其他任务。服务器官方环境为 `/root/miniconda3/envs/hyperrag_official/bin/python`，PyTorch 2.3.0+cu121。

下一步按顺序：

1. 用针对性的已有测试和少量 checkpoint 重评确认两个基准的源码与产物一致；失败则先定位，不重跑整套已完成实验。
2. 实现共享 supervision 接口、Positive Relabel、核心 toy graph 与无答案推理测试。
3. 调查并真实下载至少三个候选基准，优先接入两个适合且图可获得的基准；将选择与处理统计写入 `DATASET_SELECTION_ZH.md`。
4. 固定四策略五种子矩阵。现有结果可复用时只补缺失策略；改变训练配置时同基准所有策略一起重跑。可选 all-shortest-path 属于改变正集合的独立实验，不与 relabel 混为一谈。
5. 先产出 actual-sampler PCN prevalence，再并行训练、配对统计、可达子集及机制分析。所有新运行放独立目录，保存配置、命令、代码提交、环境、时间和 checkpoint。
6. 用 nature-writing 从头组织中文全文，再用 nature-polishing 单独润色；所有数字来自 JSON/CSV，尚未完成的结果明确 TODO，不作虚构跨基准结论。

每次检查先说明能发现的具体失败及失败后改变的行动；只运行会影响下一步的检查，不反复审计已经定论的项目。
