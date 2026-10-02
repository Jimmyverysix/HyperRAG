# 最终 Retriever-only 实验协议

## 1. 研究范围

本项目只研究一个问题：单路径弱监督是否把仍位于完整最短答案路径上的候选作为负例送入 Retriever 损失，以及只屏蔽这些冲突项能否改善检索。

实验复用 HyperRAG 官方 HyperRetriever 的 GTE 文本表示、DDE 结构编码和两层 MLP。它不是端到端 HyperRAG 生成实验，不调用生成式 LLM、付费 API、人工语义标注或 LLM Judge，也不增加 GNN、attention、teacher、learnable lambda 等模型组件。

## 2. 数据、身份与 answer-free 对齐

- 数据：WikiTopics_QE 与发布的 WikiTopicsQE_NLG，共 11 个领域。
- 结构化查询形状：`(e, (r1, r2, r3))`。
- 主题实体：只取结构化查询中的 `e`。
- 实体身份：始终使用 Wikidata QID；英文标签只作为文本编码输入，同名实体不合并。
- NLG 对齐：只使用发布顺序与主题标签代价做单调对齐，不读取 NLG 或结构化答案标签。
- 答案允许用途：训练正路径、路径一致判定、评价。
- 答案禁止用途：候选生成、GTE 相似度、DDE、MLP 特征、推理分数。

历史正式结果曾使用答案数量/答案标签辅助重建 NLG—结构化查询映射，并允许答案标签证据决定查询是否纳入。最终审计认定这会使答案间接影响 query embedding 与候选池，因此旧结果仅保留为 provenance；最终结果在独立 run root 中按 answer-free 对齐完整重跑。

## 3. 确定性传导式超图

图由 `train_graph.txt` 与 `test_inference.txt` 合并构成。后者是所有方法共享的固定检索语料，不包含测试问题或测试答案标签。相同头实体的外向事实按稳定顺序聚合为一个事实节点，形成实体—事实二部 incidence graph。

传导设置满足：

- 测试问题不参与训练；
- 测试答案标签不参与建图；
- 测试答案不参与候选生成或特征；
- 所有方法共享完全相同的图、候选和编码。

因此本文称其为“在确定性 WikiTopics 超图上复用官方 HyperRetriever 组件的 Retriever-only 受控实验”，不声称完整复现端到端 HyperRAG。

## 4. 训练候选与路径一致负例

对每个可达的主题—答案对，原流程选择一条最短路径，将所选路径转移的并集作为正例；再在同一路径引导子图中按发布采样逻辑采样等量负例，最多尝试 `20 × |positives|` 次。

对实际采样负例 `t=(v,f,u)`，若存在主题 `s` 和正确答案 `a` 满足

```text
d_I(s,v) + 2 + d_I(u,a) = d_I(s,a)
```

则 `t` 是路径一致负例。判据必须使用完整主题—答案最短路径条件，不能用“局部更靠近答案”替代。该属性是结构成员资格，不是语义真值。

## 5. 三个主策略

1. **策略1：Baseline**。所有候选权重为 1。
2. **策略2：Matched Random Masking**。对每个问题，从全部已采样负例中均匀无放回选择恰好 `|D_q|` 个并屏蔽；抽样不查看路径一致身份。
3. **策略3：Path-Consistent Negative Masking**。只屏蔽 `D_q`。

三个策略共享候选池、GTE、DDE、MLP、Adam、候选级内部训练/早停划分、初始化协议、随机种子、早停预算与推理候选。策略2用于排除“只因为少惩罚相同数量负例”的解释。

统一分析目标为

```text
w_i(lambda) = lambda, i in D_q
              1,      otherwise

L = sum_i w_i BCEWithLogits(z_i, y_i) / sum_i w_i
```

其中 `lambda=1.00` 与 Baseline 的 loss/gradient 等价，`lambda=0.00` 与删除路径一致负例后的均值 loss/gradient 等价。若一个 batch 的权重和为零，则跳过该 batch 的反向传播与 Adam step。

冻结网格为 `{0.00, 0.10, 0.25, 0.50, 0.75, 1.00}`。网格只用于验证集分析；最终主方法是否简化为固定 masking 由验证集领域等权结果与事后简化比较共同决定。

## 6. 固定候选接口

验证与测试候选只输入自然语言问题、主题实体和固定超图。每跳按问题与 head、事实节点、tail 的冻结 GTE 余弦相似度构造语义 beam，最多三跳。正式宽度为 32；它在正式 lambda sweep 前仅用 art/valid 与宽度 10 比较后冻结，未继续搜索更宽 beam。

候选生成不读取答案，不读取训练后 MLP，不做 logit 阈值截断。所有方法对同一候选池输出原始 MLP logit，分数并列时按转移字典序稳定排序。

## 7. 指标

主指标为 Answer-Path Completion MRR（APC-MRR）。对问题 `q`，令 `P_q` 为固定候选池中的完整主题—正确答案有向路径集合，定义

```text
r_q = min_{P in P_q} max_{t in P} rank(t)
```

若 `P_q` 为空，则 `RR_q=0`；否则 `RR_q=1/r_q`。APC-MRR 是问题级 `RR_q` 的均值。实现通过逐个加入 top-k 转移并记录首次形成完整答案路径的 `k` 计算；自动 toy test 验证它与 min–max 定义等价。

APC-MRR 不是传统实体排序 MRR。次指标为 Reach@10；Reach@5 只作附录诊断。

Candidate Oracle Reach 在整个固定候选池上计算：若池中存在至少一条完整主题—答案路径则为 1，否则为 0。它只诊断候选生成上限，不训练模型、不修改 beam，也不产生新的研究问题。

## 8. 选择、测试与事后简化

每个领域的 `lambda_D*` 只使用官方 valid split、五个共享种子的 APC-MRR 算术均值选择；精确并列时选择更大的 lambda。选择器拒绝 test report，测试候选准备与评价要求读取已冻结的选择文件。

另对每个固定 lambda 计算 valid 的领域等权宏 APC-MRR。若全局最优为 `lambda=0.00`，才在现有 test 上运行固定 masking。由于项目在提出简化问题前已经访问过一次 test，该步骤必须标记为 **POST-HOC SIMPLIFICATION ANALYSIS**，并同时比较：Baseline、Matched Random Masking、Fixed Path-Consistent Masking 与原领域级 tuned strategy。

## 9. 统计

- 训练种子：42、43、44、45、46。
- 配对单位：同一领域、同一 query key。
- 先在同一问题上等权平均五个种子，再进行 10,000 次 paired bootstrap。
- bootstrap seed：20260928。
- 报告：百分点差值与 95% CI。
- 跨领域：领域等权宏平均，不按查询数加权。
- 不把同一问题的不同种子当作独立样本。

## 10. 结果链与历史保护

最终结果链为：

```text
raw run reports
→ aggregate JSON/CSV
→ generated LaTeX macros/tables and PDF/SVG figures
→ paper/main.pdf
→ review_bundle/
```

旧的结构代理、旧 Retriever-only 结果和已删除的探索性支线均保持只读，不覆盖、不删除，也不进入最终论文。最终 answer-free 重跑写入独立服务器 run root；仓库中的新聚合产物统一写入 `artifacts/final_revision/`。
