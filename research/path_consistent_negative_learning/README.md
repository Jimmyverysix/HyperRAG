# HyperRAG 路径监督研究

本目录实现提案中的路径监督研究。核心问题是：公开流程只选择一条最短路径作为正例时，未被选中、但仍属于其他主题—答案最短路径的转移，应当标负、忽略，还是标正。

## 实验策略

- 策略1：保留公开基线标签，选中路径之外的采样候选标负。
- 策略2：选中路径正例保持不变；其他最短路径上的候选不参与损失。
- 策略3：任一主题—答案最短路径上的候选标正。
- 随机丢弃对照：每题随机屏蔽与策略2相同数量的负例，用于区分路径信息与样本数变化。

四个实验臂共享同一候选顺序和特征对象，只改变标签与损失掩码。主判据为

```text
d(topic, head) + 2 + d(tail, answer) == d(topic, answer)
```

其中一次“实体—超边—实体”转移在关联图中的长度为 2。较宽松的局部答案降距判据只作为消融项。

## 代码结构

- `schema.py`：不可变候选批次、标签和损失掩码数据结构。
- `distances.py`：主题—答案成对最短路径判定与局部降距消融。
- `strategies.py`：策略1、策略2、策略3及随机丢弃对照。
- `wikitopics.py`：官方 WikiTopics_QE 结构化数据加载与候选图复现。
- `audit_wikitopics.py`：发生率审计、五随机种子采样模拟和分层统计。
- `metrics.py`：MRR、Hits@K、Recall@K 和 PR-AUC。
- `statistics.py`：严格按问题配对的自助法区间和多随机种子汇总。
- `structured_data.py`：门槛 C 结构化代理实验的固定候选、查询级划分与四臂标签。
- `structured_model.py`：共享实体/关系嵌入和 MLP 检索器。
- `train_structured.py`：独立运行目录、早停、检索指标和答案可达率评测。
- `run_weighted_suite.py`：策略2加权实验的六卡任务调度与冻结配置校验。
- `select_weight.py`、`confirm_weight.py`：专用选参和独立确认的自动判定。
- `prepare_gate_d.py`、`run_gate_d_all.py`：确认通过后才可启动的跨领域门槛 D 编排。
- `tools/upload_artifact.py`：仅用于向服务器断点续传数据或模型；代码同步仍使用 GitHub。

## 运行结构审计

```powershell
python -m research.path_consistent_negative_learning.audit_wikitopics `
  --dataset-root /path/to/WikiTopics_QE `
  --output-dir runs/wikitopics_audit `
  --seeds 42 43 44 45 46
```

命令只读取官方 `train_graph.txt`、`train_queries.pkl`、`train_answers_hard.pkl` 和 `og_mappings.pkl`。输出包含逐题 JSONL、各领域汇总、候选池发生率、实际采样发生率以及按跳数、超边元数和候选深度的分层统计。由于作者没有发布最终 GraphML，该审计按照公开构图规则从官方整数图执行确定性结构重建；等长路径使用固定字典序择一，因此它不是作者原运行产物的字节级复原。

## 测试

```powershell
python -m unittest discover -s research/path_consistent_negative_learning/tests -v
```

测试覆盖等长替代路径、多主题多答案、不连通节点、有向图、高元超边、四实验臂共享候选、随机对照复现性、指标和配对统计。

## GitHub 同步约定

本地与服务器只通过 GitHub 研究分支同步代码：

```text
research/path-consistent-negative-learning
```

训练数据、模型权重和运行日志不提交到 GitHub，保存在服务器独立运行目录；聚合后的 JSON、CSV、论文图和最终 PDF 才进入研究产物目录。服务器更新代码时只允许快进合并，避免覆盖本地或服务器上的未提交修改。

## 门槛 C 结构化代理实验

官方仓库没有发布建图后的 GraphML，服务器也没有原建图流程所需的大模型缓存。因此，门槛 C 使用官方整数 ID 图和 `kg2text.py` 的按 head 聚合规则构造结构代理；它不冒充原版端到端 HyperRAG 复现。候选、查询划分和特征在四个实验臂间固定，只改变标签或损失掩码。

```powershell
python -m research.path_consistent_negative_learning.prepare_structured_experiment `
  --domain-dir /path/to/WikiTopics_QE/edu `
  --sampler-seed 42 `
  --output runs/gate_c/edu/seed_42/data.pt

python -m research.path_consistent_negative_learning.train_structured `
  --data runs/gate_c/edu/seed_42/data.pt `
  --strategy strategy2_ignore `
  --seed 42 `
  --output-dir runs/gate_c/edu/strategy2_ignore/seed_42
```

训练、验证和测试按问题划分，避免同一问题的候选三元组同时出现在不同划分中。固定评测同时报告公开选中路径、全部最短路径以及 Top-K 转移能否从主题实体到达答案。

六卡运行四个实验臂和五个共享随机种子：

```powershell
python -m research.path_consistent_negative_learning.run_structured_suite `
  --data-dir runs/gate_c/edu/datasets `
  --output-dir runs/gate_c/edu/runs `
  --seeds 42 43 44 45 46 `
  --gpus 0 1 2 3 4 5
```

调度器每张卡只启动一个训练进程，并拒绝超过 6 张卡的配置。完成后分别聚合门槛 B 和门槛 C：

```powershell
python -m research.path_consistent_negative_learning.aggregate_audit `
  --input-dir runs/wikitopics_audit `
  --output-dir runs/wikitopics_audit/combined

python -m research.path_consistent_negative_learning.aggregate_training `
  --run-root runs/gate_c/edu/runs `
  --output-dir runs/gate_c/edu/summary
```

## 生成论文结果

聚合文件复制到 `artifacts` 后，图表与论文数字均由脚本生成：

```powershell
python -m research.path_consistent_negative_learning.figures.gen_fig_prevalence `
  --report research/path_consistent_negative_learning/artifacts/audit/combined_summary.json `
  --output-dir research/path_consistent_negative_learning/artifacts/figures

python -m research.path_consistent_negative_learning.figures.gen_fig_gate_c `
  --summary research/path_consistent_negative_learning/artifacts/gate_c/training_summary.json `
  --output-dir research/path_consistent_negative_learning/artifacts/figures

python -m research.path_consistent_negative_learning.figures.gen_fig_weighted `
  --summary research/path_consistent_negative_learning/artifacts/weighted/selection_summary.json `
  --output-dir research/path_consistent_negative_learning/artifacts/figures

python -m research.path_consistent_negative_learning.proposal.generate_results_tex `
  --audit research/path_consistent_negative_learning/artifacts/audit/combined_summary.json `
  --training research/path_consistent_negative_learning/artifacts/gate_c/training_summary.json `
  --selection research/path_consistent_negative_learning/artifacts/weighted/selection_summary.json `
  --confirmation research/path_consistent_negative_learning/artifacts/weighted/confirmation_summary.json `
  --gate-d research/path_consistent_negative_learning/artifacts/weighted/gate_d_summary.json `
  --output research/path_consistent_negative_learning/proposal/generated_results.tex
```

若选参没有合格权重，省略 `--confirmation` 和 `--gate-d`；生成器会把后续阶段明确写为“未运行”。唯一主文件是 `proposal/main.tex`，在该目录编译后输出 `proposal/main.pdf`。不再另设实验报告文档。

## 策略2加权改进

原策略2在教育领域把争议负例完全移出损失，答案可达率提高，但选中路径曲线下面积下降 1.32 个百分点，超过结果产生前固定的 1 个百分点保护线。后续阶段不改写这项失败判定，而是把争议负例权重设为 $\lambda\in[0,1]$：$\lambda=0$ 等价于原策略2，$\lambda=1$ 等价于策略1。随机对照按题随机选择等量负例并赋予相同权重。

完整协议冻结在 `protocols/weighted_strategy2.json`。教育领域按 `60/10/15/15` 划分训练、早停、选参和保留测试部分；权重只在专门的选参部分确定。调度命令为：

```powershell
python -m research.path_consistent_negative_learning.run_weighted_suite `
  --phase selection `
  --data-dir runs/weighted/selection/edu/datasets `
  --output-dir runs/weighted/selection/edu/runs `
  --seeds 42 43 44 45 46 `
  --strategy2-weights 0 0.1 0.25 0.5 0.75 1 `
  --random-control-weights 0.1 0.25 0.5 0.75 `
  --include-endpoint-reference `
  --evaluation-split selection `
  --gpus 0 1 2 3 4 5

python -m research.path_consistent_negative_learning.select_weight `
  --run-root runs/weighted/selection/edu/runs `
  --output-dir runs/weighted/selection/edu/summary
```

只有选参条件满足且两个端点等价检查通过，才锁定唯一权重。独立确认预先固定在未参与调参的获奖领域，使用新种子 142--146；确认调度器会读取锁定文件，并拒绝领域、种子、数据划分、权重或训练超参数与冻结协议不一致的运行。确认同时要求策略2加权版相对策略1和随机降权对照的答案可达率区间下界大于零，以及选中路径曲线下面积的一侧 95% 下界高于 $-1$ 个百分点。只有三项同时满足才进入门槛 D。
