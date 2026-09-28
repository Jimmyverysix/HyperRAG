# Retriever-only 零 LLM 正式实验协议

冻结日期：2026-09-28

状态：`frozen_before_retriever_only_runs`

配置真源：`configs/retriever_only/wiki_main.json`

## 1. 研究范围

本协议检验：单路径弱监督是否对“未被选中、但位于另一条完整 topic-answer 最短路径上的实际采样负例”施加了过强负监督。

正式实验使用 HyperRAG 官方 HyperRetriever 的 GTE 文本表示、DDE 结构编码、两层 MLP 和训练超参数，并在确定性重建的 WikiTopics 超图上执行 Retriever-only 训练与评价。全流程不调用生成式 LLM，不调用 OpenAI、百炼或其他付费 API，不生成答案文本，不使用人工标注或 LLM Judge。

历史结构代理结果只作为 Structural Diagnostic Study。新实验不得覆盖历史 artifacts，也不得表述为“完整复现官方 HyperRAG”。

## 2. 数据与图

领域固定为：art、award、edu、health、infra、loc、org、people、sci、sport、tax。

数据使用 WikiTopics_QE 整数图和现有 WikiTopics NLG。整数图经 `og_mappings.pkl` 映射到 Wikidata ID；`train_graph.txt` 与 `test_inference.txt` 合并后，按 head 实体聚合外向事实为超边。实体文本来自冻结英文标签快照；超边文本由同一超边内的结构化事实按稳定顺序序列化为 `head | relation: tail; ...`，不调用生成式模型，也不使用无法与结构化 head 可靠回溯对齐的历史 graph sentences。自然语言 query 来自现有 NLG query 文件。

topic entity 直接取结构化 query 的首实体 `e`。hard answer 来自 benchmark ground truth，只用于最短路径监督与评价，禁止进入候选生成、文本编码、DDE 或 MLP scoring。

NLG 与结构 query 的对齐必须复现原转换脚本的标签过滤，并对每个 domain/split 做精确数量断言。任何不一致都在训练前失败。

## 3. 训练 baseline

对每个可达 topic-answer pair 选择一条最短路径，将所选路径转移的并集作为 positives。围绕这些路径构造官方 path-guided candidate subgraph，再按官方 released sampler 从中采样与 positives 等量的 negatives，最多尝试 `20 x |positives|` 次。

主实验的最短路径按稳定字典序选择。采样、模型初始化、候选级 80/20 stratified internal split 和 DataLoader 共享种子 42、43、44、45、46。Baseline、Matched Random 与路径方法在同一个 seed 下共享候选、特征、划分、初始化协议与训练预算。

候选表示固定为：query GTE、head GTE、hyperedge GTE、tail GTE 与 30 维 DDE 的拼接。MLP 固定为 `4126 -> 256 -> 1`，激活为 ReLU。训练固定为 Adam、学习率 `0.0001`、batch size 32、最多 50 轮、patience 10、`min_delta=0.00001`。

## 4. 路径一致负例与损失

只对已经被 baseline 采样为 negative 的候选 `t=(v,f,u)` 判断。若存在 topic `s` 和正确 answer `a` 满足

```text
d_I(s,v) + 2 + d_I(u,a) = d_I(s,a)
```

则它是路径一致负例。该定义只表示候选属于另一条完整最短答案路径，不表示它一定是语义正例或人工确认的有效证据。

逐样本权重为：路径一致负例取 `lambda`，其他样本取 `1.00`。损失固定为逐样本 `BCEWithLogits` 的加权和除以当前 batch 的权重和。网格固定为 `{0.00, 0.10, 0.25, 0.50, 0.75, 1.00}`；`1.00` 是 Baseline，`0.00` 是 Masking，中间值是 Soft Weighting。不得追加事后网格点。

## 5. Retrieve-only 候选接口

验证与测试只输入自然语言 query、topic entity 和确定性超图。直接三跳全展开在 art 的实测典型规模约为每题 1,560 万候选，因此冻结为答案不可见的 GTE 语义束接口：每跳先按冻结 GTE 的 query 与 head、hyperedge、tail 平均余弦相似度保留 10 个转移，最多三跳；反向重复转移按稳定顺序去重。该候选束不读取 hard answer，不读取训练后 MLP，对所有方法、lambda 和种子完全相同。DDE 按官方单跳候选接口计算；接口直接返回 candidate transition 与原始 MLP logit，不做 logit 阈值截断，不调用生成器。

## 6. 指标

对每个 query，将候选按 logit 降序排序；同分时按 transition 的稳定字典序排序。令 `r_q` 为最小的 `k`，使 top-k 有向转移构成的子图能够从 topic 到达任一正确 hard answer。若不存在则 `RR_q=0`，否则 `RR_q=1/r_q`。

主指标：Answer-Path MRR。

次指标：Answer Reach@10；补充 Answer Reach@5。

selected-path PR-AUC 等旧指标仅保留为历史 Appendix diagnostic，不作为本轮主指标或选择条件。

## 7. lambda 选择与方法决策

每个领域只在官方 valid split 上，以五个共享种子的 Answer-Path MRR 算术均值选择 `lambda_D*`。完全并列时选择更大的 lambda。选择器拒绝 test 输入；test 运行只能读取冻结的选择文件。

11 领域正式 sweep 完成后生成 `docs/METHOD_DECISION.md`：若中间 lambda 在多个领域稳定优于 `0.00`，最终方法保留 Soft Weighting；若绝大多数领域选择 `0.00` 且中间 lambda 没有稳定额外收益，则按奥卡姆剃刀简化为 Masking。不得预设结论。

## 8. 三个正式实验部分

### A. Conflict Prevalence

只报告实际 sampled conflict rate、affected-query rate 和 11-domain 分布。

### B. Main Retriever Experiment

最终表只比较：1. Baseline；2. Matched Random；3. Ours。

Matched Random 对每个 query 从普通 negatives 中随机选取恰好 `|D_q|` 个样本。若最终方法是 Masking，则屏蔽相同数量；若最终方法是 Soft Weighting，则赋予相同 lambda。其他设置全部一致。

### C. Path-Selection Sensitivity

只在存在多条等长最短路径的训练 query 上改变最短路径择一。固定变体种子 2718、3141、5772，使用 shortest-path DAG 和 seeded predecessor ordering 为每个 topic-answer pair 选一条路径，不枚举全部路径。每个变体训练 Baseline 与 Ours，报告 Answer-Path MRR、Answer Reach@10 的 mean、standard deviation 和 range。

P0 完成前不开发第二数据集或新模型。

## 9. 统计

主要比较按 query key 配对。先对同一 query 的五个 seeds 等权平均，再执行 10,000 次 paired bootstrap，bootstrap seed 为 20260928，报告百分点差与 95% CI。不得把同一 query 的多个 seed 当作独立样本。

跨领域使用 equal-domain macro average。lambda 选择只看 valid；所有统计脚本必须拒绝选择阶段的 test 数据。

## 10. 测试门槛

正式 sweep 前必须通过：

- 等长最短路径候选被识别；
- 非最短路径不被识别；
- 局部降距成立但全局条件不成立的反例；
- `lambda=1.00` 与 baseline loss/gradient 等价；
- `lambda=0.00` 与删除路径一致负例后求均值等价；
- Matched Random 每个 query 数量严格相同；
- 相同 seed 的路径选择、采样和训练数据可复现；
- NLG/结构 query 单调对齐完整；当前 Wikidata 已缺少英文标签的 topic/answer 单独计数并排除，不静默丢弃；
- 答案实体不进入 inference feature；
- test split 不参与 lambda selection。

## 11. 结果与 provenance

每个 run 保存：研究代码 commit 与 dirty state、官方上游 commit、config、domain、split、seed、lambda、method、完整 command、环境、GPU、指标、checkpoint 和逐 query 结果。中断重跑不得覆盖元数据不一致的目录。

结果链固定为：

```text
raw artifacts -> aggregate JSON/CSV -> LaTeX macros/tables -> PDF/SVG figures -> paper
```

禁止手工抄写实验结果进 LaTeX。所有小于 1 的小数必须有前导 0。
