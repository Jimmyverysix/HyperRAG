# WWW 2027 路径监督研究实施计划

## 1. 不可变约束

- 历史 artifacts、历史 proposal 和历史 PDF 只读。
- 新结果只写入 `artifacts/www_revision/`。
- 主方法只有一个新超参数 $\lambda$；不增加模型、教师、风险估计器或推理模块。
- 主定义只使用完整主题—答案最短路径条件。
- $\lambda=1$ 必须数值等价官方 baseline；$\lambda=0$ 必须数值等价完全屏蔽路径一致负例。
- baseline、ours 和匹配随机对照共享数据划分、候选池、训练种子与初始化协议。
- validation 只选超参数和早停；test 在 $\lambda$ 冻结前不可访问。
- 所有论文数字从 raw/aggregated artifacts 自动生成。

## 2. 目录规划

```text
research/path_consistent_negative_learning/
├── path_supervision/              # 正式可复用核心模块
│   ├── path_consistency.py
│   ├── weighted_loss.py
│   ├── config.py
│   ├── lambda_selection.py
│   ├── provenance.py
│   └── semantic_audit.py
├── scripts/                       # 可复现 CLI
│   ├── verify_historical_artifacts.py
│   ├── run_official_experiment.py
│   ├── run_experiment_queue.py
│   ├── select_lambda.py
│   ├── sample_semantic_audit.py
│   ├── analyze_semantic_audit.py
│   └── generate_paper_results.py
├── configs/www_revision/          # 冻结协议与实验配置
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── METHOD_SPEC.md
│   ├── REPRODUCIBILITY.md
│   └── SEMANTIC_AUDIT_GUIDE.md
├── artifacts/www_revision/        # 本轮新聚合产物与运行清单
├── paper/www2027/                 # 单一投稿稿件与自动生成表图
├── REPO_AUDIT.md
├── IMPLEMENTATION_PLAN.md
├── CHANGELOG_WWW_REVISION.md
└── FINAL_RESEARCH_REVISION_REPORT.md
```

历史模块优先复用；只有正式接口和职责不清时才抽取，不复制已验证算法。

## 3. 阶段 P0-1：基线、资产与历史结果核验

1. 运行现有全量单元测试，记录测试数和环境。
2. 新增历史 artifacts 只读验证器，从 JSON 自动验证 11 领域、90,000 问题、候选池比例、采样比例、原始失败和历史 transfer 数值。
3. 只读检查服务器的完整 WikiTopics NLG、GraphML、GTE 表示、Retriever 检查点、开放域图和 API 配置是否存在；报告路径与可用性，不复制密钥。
4. 确认官方代码提交与本地 `origin` 状态，正式实验通过适配器调用官方候选构造、表示和 MLP，不修改 vendor 基线语义。

退出条件：历史验证器通过；baseline 输入输出边界明确；official end-to-end 可用资产与缺口形成机器可读清单。

## 4. 阶段 P0-2：正式核心实现与测试

### 4.1 路径一致性

抽取稳定 API：

```python
is_path_consistent(source_entities, answer_entities, transition, distance_index)
find_path_consistent_negatives(query, sampled_negatives, distance_index)
```

复用 `distances.py` 的成对距离等式与有向图处理；局部降距只作为显式消融接口。

### 4.2 加权目标

实现独立的 `weighted_binary_cross_entropy`：

```text
sum(weight * BCE) / sum(weight)
```

支持 AMP，标签保持不变；权重和为零时给出明确错误。

### 4.3 数据集级 $\lambda$ 选择

固定网格：`[0.00, 0.10, 0.25, 0.50, 0.75, 1.00]`。

每个 Dataset × Retriever 在自己的 validation 问题上选择 $\lambda^*$，自动输出 `lambda_star.json`。选择器只读取 validation 字段，若输入含 test 候选选择分数则拒绝。正式指标在 `EXPERIMENT_PROTOCOL.md` 冻结后不再修改。

### 4.4 必备测试

- 两条等长最短路径的 toy graph。
- 可到达但非最短路径。
- “局部降距为真、完整成对条件为假”的反例。
- $\lambda=1$ 与 baseline loss 等价。
- $\lambda=0$ 与正确归一化的 mask loss 等价。
- 匹配随机降权逐题数量相等。
- 同种子候选采样、随机对照和初始化可重复。
- lambda selector 不读取 test 指标。
- official baseline 适配前后输出等价。

退出条件：全部历史测试和新增测试通过后，才允许启动 GPU sweep。

## 5. 阶段 P0-3：正式实验协议冻结

在 `docs/EXPERIMENT_PROTOCOL.md` 中写死：

- 数据集与官方划分。
- 问题级 train/validation/test 规则。
- lambda 网格。
- 选择指标与 tie-break。
- 共享种子 42–46。
- 主要/次要指标。
- 10,000 次问题级配对 bootstrap。
- 等权领域宏平均与补充 micro 汇总。
- test 访问规则。
- 早停、训练和生成设置。

初步选择原则：优先使用官方最终任务 validation 指标；若端到端 validation 成本或资产不允许，则使用单一、预先固定的标准 Retriever 指标，绝不构造 reachability 与 PR-AUC 的人工复合分数。最终选择在服务器资产核验后冻结，且必须早于任何正式 test 运行。

## 6. 阶段 P0-4：结构审计和代理主实验

### RQ1：冲突是否真实且进入损失

- 重新从 raw 数据生成候选池与实际采样统计。
- 输出逐域 JSON/CSV、per-query 分布、至少一个冲突问题比例、均值/中位数/四分位数、macro 和 micro。

### RQ2：硬决策与软监督

实验臂：

- baseline：$\lambda=1$。
- mask endpoint：$\lambda=0$。
- ours：dataset-specific $\lambda_D^*$。
- fixed transfer：历史 $\lambda=0.1$。
- matched random reweighting。
- all-shortest-path positive 诊断对照。

### RQ3/RQ4：软监督与参数敏感性

- 每个数据集跑固定 lambda 网格。
- validation 自动选择后仅运行冻结 test。
- 报告 mean ± std、配对差值与 95% 区间。
- 探索性报告 sampled conflict rate 与 $\lambda_D^*$ 的 Spearman 相关，不因无相关而删除。

退出条件：raw 运行完整、聚合脚本可从空目录重建所有新表图、无 test 泄漏。

## 7. 阶段 P1：官方 Retriever 与端到端 QA

优先顺序：

1. WikiTopics 11 领域正式 HyperRetriever。
2. 资产最完整的一个开放域数据集；优先 HotpotQA，若官方图或缓存更完整则按事实调整。
3. 计算允许时再扩展 2WikiMultiHopQA 和 MuSiQue。

每个实验保存：experiment ID、时间、Git 提交、dirty state、命令、配置、数据版本、种子、lambda、backbone、CUDA 设备、Python/PyTorch/CUDA 版本、validation/test 指标和 checkpoint pointer。

生成器固定、prompt 固定、解码固定，baseline 与 ours 共用。没有真实 QA 输出时，论文只报告 Retriever/代理结果。

## 8. 阶段 P1：语义人工审计准备

- 分层随机抽取 300–500 条路径一致负例。
- 输出盲化 CSV/JSONL，不显示策略、lambda 或模型分数。
- 标签只有：有用证据、结构可达但语义不足、无法判断。
- 实现双标注者合并、分歧裁决和 Cohen’s kappa 分析。
- 没有两名真实标注者时，结果明确写为 `PENDING HUMAN AUDIT`。

## 9. 图表、论文和自动同步

使用 Python/Matplotlib，统一 Okabe–Ito 色盲安全语义；输出 PDF + SVG，PNG 预览至少 300 dpi。

- 图1：selected path、未选中的路径一致 path、真正 off-path negative。
- 图2：11 领域候选池与实际采样冲突率。
- 图3：lambda sensitivity，分别显示主性能、coverage 和 ranking，不使用双 y 轴。
- 图4：逐数据集/领域差值与 95% 区间 forest plot。
- 图5：$\lambda_D^*$ 与冲突率的 Spearman 探索性关系。

`scripts/generate_paper_results.py` 从聚合 JSON 生成 `paper/www2027/generated/results.tex`、表格和图。投稿主文按 Introduction、Related Work、Path Supervision Conflict、Confidence-Weighted Negative Supervision、Experiments、Analysis/Discussion、Limitations、Conclusion 组织，不再以 Gate 或策略编号组织。

## 10. 复现入口

提供 Makefile 或等价入口：

```text
make audit
make sweep
make main-experiments
make figures
make paper
```

服务器训练代码只通过 GitHub 分支同步；大型数据、checkpoint 和 raw runs 留在服务器，仓库只提交协议、聚合结果、图表、论文和来源记录。

## 11. 每阶段自检

代码：测试通过、baseline 不变、端点等价、确定性、无 test 泄漏。

实验：共享种子、正确划分、validation 冻结、无删 seed/dataset、区间单位正确。
论文：每项 claim 可追溯到 artifact，不把结构一致写成语义相关，不把代理写成端到端，不隐藏 $\lambda=0$ 的负结果。

## 12. 最终交付

最终生成 `FINAL_RESEARCH_REVISION_REPORT.md`，逐项列出修改、命令、历史与新结果、未完成实验、official end-to-end 状态、人工审计状态、图表/表格/PDF路径、测试结果、阻塞项和三项最值得人工继续的工作。
