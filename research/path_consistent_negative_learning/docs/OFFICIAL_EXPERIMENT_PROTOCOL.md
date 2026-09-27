# 官方 HyperRetriever 正式实验冻结协议

冻结日期：2026-09-27

状态：`frozen_before_baseline_reproduction`

配置真源：`configs/official_hyperrag/wiki_main.json`

## 1. 上游、数据与证据边界

- 官方仓库固定为 `https://github.com/Vincent-Lien/HyperRAG`。
- 官方提交固定为 `6d5a9033353c516a9220d78591f2c666f19ee0b1`。
- 主实验使用官方 WikiTopics NLG 的 11 个领域与官方 train/valid/test。
- 历史结构代理实验全部只读保留，只作为 Mechanism / Diagnostic Study，不冒充官方 HyperRetriever 或端到端 QA。
- 新产物只写入 `artifacts/official_hyperrag/`，不得覆盖历史 artifacts。

## 2. 严格执行顺序

1. 原样复现一个领域的官方 baseline，完成训练、推理和 MRR/Hit@10 评价。
2. 扩展为 11 个领域的官方 baseline，并如实记录与论文值的差异。
3. 只给实际采样负例增加 `path_consistent: bool` 元数据。
4. 接入按权重和归一化的 BCE，并通过全部端点等价测试。
5. 在官方 valid 上完成 λ 选择；选择冻结后才访问 test。
6. 根据结果决定最终方法是 Masking 还是 Soft Weighting。
7. 运行 matched random、Path-selection Sensitivity 和一个开放域 benchmark。
8. 最后重写论文、主图和 `FINAL_WWW_STATUS.md`。

baseline 未跑通前，不允许把监督补丁接入正式训练，也不启动大规模 GPU sweep。

## 3. WikiTopics 主实验

领域：art、award、edu、health、infra、loc、org、people、sci、sport、tax。

共享随机种子：42、43、44、45、46。

每个任务使用单张 GPU，只允许 `CUDA_VISIBLE_DEVICES=0` 至 `5`，不使用 DDP。

除下列权重外，候选集合、正负标签、GTE 表示、DDE、两层 MLP、Adam、学习率、batch size、训练轮数、早停、推理和 evaluator 均与已复现的 baseline 相同：

\[
w_i(\lambda)=
\begin{cases}
\lambda,& i\text{ 是路径一致负例},\\
1,& \text{其他样本},
\end{cases}
\qquad
\mathcal L=\frac{\sum_i w_i\,\mathrm{BCEWithLogits}(z_i,y_i)}{\sum_i w_i}.
\]

固定搜索空间为 `{0.00, 0.10, 0.25, 0.50, 0.75, 1.00}`，不追加事后网格点：

- `1.00`：官方 baseline；
- `0.00`：Path-Consistent Negative Masking；
- `0.10`、`0.25`、`0.50`、`0.75`：Soft Weighting。

对每个领域，以五个种子的官方 valid MRR 均值选择 λ。完全并列时选择更大的 λ。test 任务必须读取冻结的选择文件，不能通过命令行临时指定 λ。正式报告 test MRR、Hit@10 与逐领域/宏平均的配对差值；置信区间以问题为重采样单位。

## 4. 路径一致判据

只对官方流程已经采样为 negative 的候选 `t=(v,e,u)` 判断。定义主题实体集合为 `S_q`、答案实体集合为 `A_q`。若存在 `(s,a) ∈ S_q × A_q` 满足

\[
d_I(s,v)+2+d_I(u,a)=d_I(s,a),
\]

则标记 `path_consistent=true`。实现只计算距离，不显式枚举全部最短路径。局部“尾实体更接近答案”不能替代该定义。

## 5. 必须通过的自动测试

1. `lambda=1.00` 的 loss 与 gradient 和官方 `BCEWithLogitsLoss` 等价。
2. `lambda=0.00` 严格等价于删除路径一致负例后，对剩余样本求均值。
3. 完整成对距离条件能够正确判断路径一致性。
4. 局部降距但不属于完整最短路径的反例必须判为 false。
5. 同一随机种子的负例采样结果可复现。
6. λ 选择器拒绝 test split。

## 6. 方法决策

正式 11 领域结果完成后才生成 `docs/METHOD_DECISION.md`：

- 若多个领域中间 λ 稳定优于 `0.00`，保留 Soft Weighting；
- 若绝大多数领域选择 `0.00`，且中间 λ 没有稳定优势，最终方法简化为 Masking。

决定以正式语义 Retriever 为准，不以历史代理实验或旧标题为准。

## 7. 唯一主要机制对照

Matched Random Control 按问题从普通负例中抽取恰好 `|D_q|` 个样本，其中 `D_q` 是路径一致负例集合，并赋予和最终方法相同的权重。抽样种子与主实验共享；其他设置全部不变。该对照只回答“收益是否来自路径定位，而不是单纯减少负监督”。

## 8. Path-selection Sensitivity

固定三个路径选择变体，种子为 2718、3141、5772。每个变体先按稳定排序重建同一张图，再用对应局部随机数生成器打乱节点与边的插入顺序；图拓扑不变，只改变等长最短路径的 tie-breaking。

每个变体分别训练 official baseline 和最终 path-aware 方法，使用相同训练种子与预算。报告 MRR/Hit@10 的 mean、std 和 range，检验平均性能是否提高、对路径任选的敏感性是否降低。不增加网络模块。

## 9. 开放域 benchmark

冻结数据集为 `2wikimultihopqa`。选择理由是三个发布流程的完整度相同，而它的语料规模最小、首次构图成本最低，并有官方论文参照值。

发布数据只有 train/test。固定从 800 个 train 问题中按种子 20260927 做问题级 80%/20% train/validation 划分；test 原样冻结。λ 仍从六点网格中按 validation F1 选择，完全并列时选择更大的 λ。最终只报告 test EM 与 F1，并比较 baseline、matched random 和最终方法。

## 10. Provenance 与产物

每个正式 run 保存：`experiment_id`、研究仓库 commit 与 dirty state、官方上游 commit、完整命令、配置、数据集、split、seed、λ、GPU、Python、PyTorch、CUDA、指标、checkpoint 和 predictions。

建议目录：

```text
artifacts/official_hyperrag/
  preflight/
  baseline/
  wiki_main/
  matched_random/
  path_selection_sensitivity/
  open_domain_2wiki/
  selections/
  summaries/
```

任何中断重跑必须使用新的 `experiment_id`，不得覆盖已有运行目录。
