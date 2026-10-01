# 最终实验正确性审计

## 审计结论

审计发现并修正了一项会影响正式声明的真实问题：历史 NLG—结构化查询对齐把答案数量和答案标签纳入匹配代价，并允许答案标签证据决定查询是否进入 Retriever 数据。候选生成函数本身虽然没有 `answers` 参数，但答案会通过“问题文本对应哪个结构化主题”的映射间接影响 query embedding 和候选池，因而不满足“答案不得进入候选生成/GTE”的硬约束。

修正后，对齐只使用发布顺序和主题标签代价；答案只在完成对齐后用于训练路径、路径一致判定和评价。历史结果没有被覆盖，而是在新的 run root 中完整重做候选准备、validation sweep、test 与事后固定 masking。最终论文和 `review_bundle` 只读取修正版聚合产物。

除该问题外，未发现需要改变研究设计或扩大研究范围的正确性错误。

## 1. Provenance 边界

- 历史正式结果提交：`4c91095`，只读保留。
- answer-free 修正起始提交：`5d9f4a7`。
- 修正版分支：`codex/www-final-revision`。
- 修正版服务器 run root：`/root/hyperrag_pcneg/runs/final_revision_answer_free`。
- 冻结上游 HyperRAG 提交：`6d5a9033353c516a9220d78591f2c666f19ee0b1`。
- 修正版聚合目录：`research/path_consistent_negative_learning/artifacts/final_revision/`。

每个训练/评价 report 保存研究提交、dirty state、上游提交、完整命令、环境、GPU、领域、split、seed、方法、lambda、checkpoint 与逐问题指标。修正版没有写入或覆盖历史 run root。

## 2. 数据与泄漏审计

### A. Topic entity 来源

**通过。** Topic entity 只由结构化查询 `(e,(r1,r2,r3))` 中的 `e` 经 `og_mappings.pkl` 转为 Wikidata QID。自然语言问题不负责识别 topic entity。

### B. Answer 使用范围

**历史版本不通过，修正版通过。**

历史问题如本报告开头所述。修正版中：

- `align_queries()` 只接收 topic labels 与 NLG questions；
- `load_aligned_queries()` 在映射冻结后才读取结构化 answer IDs；
- `semantic_beam_candidates()` 只接收 graph、topic、冻结 node score、beam width 与最大跳数；
- GTE 候选得分只使用 query 与 head/fact/tail 文本表示；
- DDE 只接收 candidate transition 与 topic；
- MLP 输入固定为 query/head/fact/tail GTE 与 DDE；
- evaluator 才读取 answers 计算 APC-MRR、Reach@5/10 与 Candidate Oracle。

答案是否存在于图中仍用于判断问题能否被监督和评价，但答案身份不改变该问题的候选池内容。

### C. 实体身份

**通过。** 图节点身份使用 Wikidata QID；事实节点使用带 `H:` 前缀的 owner QID。英文标签只进入编码文本。测试包含“不同 QID 具有相同英文标签”的反例，确认不会合并。

### D. 三个策略是否只改变损失权重

**通过。** 同一 seed 的策略1/2/3读取同一训练 `PreparedCandidates`、同一 embedding store、同一候选级内部划分与同一初始化协议。`method_weights()` 是方法间唯一的数据路径分支；评价阶段共享同一个固定 test candidate store。

### E. `lambda=1.00` 等价性

**通过。** 自动测试同时比较权重张量、loss 与 logits gradient，确认与普通 mean BCE 一致。

### F. `lambda=0.00` 等价性

**通过。** 自动测试确认零权重候选与显式删除这些候选后求均值的 loss/gradient 一致。全零 batch 不执行 Adam step，避免零梯度下动量仍更新参数。

### G. Matched Random 数量

**通过。** 每个 query 从全部已采样负例均匀无放回抽取恰好 `|D_q|` 个。即使该 query 的全部负例都属于 `D_q`，定义仍成立。随机集合允许偶然与 `D_q` 重合，因此是偏保守对照。

### H. Test 隔离

**通过。**

- lambda selector 只接受 `split=valid` 且 `method=ours` 的 report；
- selection artifact 明确记录 `test_metrics_accessed=false`；
- test candidate preparation 和 test evaluation 都要求冻结选择文件；
- beam 只在 art/valid 比较 10 与 32 后冻结；
- early stopping 使用训练候选内部的分层 80/20 划分，不读取 benchmark valid/test。

## 3. Transductive graph 审计

**通过。** 图由训练图和发布的 test inference context 组成，但不读取 test questions 或 test answer labels。test context 是所有方法共享的固定检索语料。测试答案不进入图构造、候选生成、特征或 score。论文已明确说明这一范围，避免把 transductive corpus 误解为 test-label leakage。

## 4. APC-MRR 审计

最终名称为 **Answer-Path Completion MRR（APC-MRR）**。对完整路径集合 `P_q`：

```text
r_q = min_{P in P_q} max_{t in P} rank(t)
RR_q = 0             if P_q is empty
       1 / r_q       otherwise
```

生产实现以 top-k 转移递增构图并记录第一次连通正确答案的 `k`。独立 toy test 用 bottleneck shortest-path 算法计算上述 min–max 形式，确认两种定义完全一致；候选池没有完整路径时返回零。

## 5. 测试证据

最终测试集覆盖以下必需断言：

1. 等长最短路径上的候选被识别；
2. 非最短路径候选不被识别；
3. 仅局部距离下降但不满足完整等式的候选不被识别；
4. `lambda=1.00` 与 baseline loss/gradient 等价；
5. `lambda=0.00` 与 hard masking loss/gradient 等价；
6. Matched Random 对每个 query 的数量严格匹配；
7. 全部负例均冲突时 Matched Random 仍有定义；
8. 相同 seed 的路径选择、采样和训练顺序可复现；
9. NLG/结构化 query 单调对齐消费全部 NLG 问题；
10. 当前标签缺失时有显式计数和过滤；
11. 删除 NLG answer 文件后仍能完成 query 对齐；
12. selector 拒绝 test report；
13. APC-MRR 的递增 top-k 与 min–max 定义一致。

最终全项目测试结果由仓库 CI 风格命令 `python -m pytest -q research/path_consistent_negative_learning/tests` 产生；具体通过数量在最终交付时记录，不把重复跑同一检查当作额外证据。

## 6. 审计后的结果使用规则

- 旧 `artifacts/retriever_only/` 结果：仅 provenance，不进入最终论文数字。
- 新 `artifacts/final_revision/` 结果：唯一论文数值来源。
- 固定 masking test：明确标记为事后简化分析。
- 历史路径选择敏感性产物：保留只读，但不进入论文、附录或新实验计划。
- 第二数据集：本轮不追加。当前正确性问题已通过重跑处理，新增不兼容 benchmark 会引入新的监督定义，不能替代本轮审计。
