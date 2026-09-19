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
- `tools/upload_artifact.py`：仅用于向服务器断点续传数据或模型；代码同步仍使用 GitHub。

## 运行结构审计

```powershell
python -m research.path_consistent_negative_learning.audit_wikitopics `
  --dataset-root /path/to/WikiTopics_QE `
  --output-dir runs/wikitopics_audit `
  --seeds 42 43 44 45 46
```

命令只读取官方 `train_graph.txt`、`train_queries.pkl`、`train_answers_hard.pkl` 和 `og_mappings.pkl`。输出包含逐题 JSONL、各领域汇总、候选池发生率、实际采样发生率以及按跳数、超边元数和候选深度的分层统计。

## 测试

```powershell
python -m unittest discover -s research/path_consistent_negative_learning/tests -v
```

测试覆盖等长替代路径、多主题多答案、不连通节点、有向图、高元超边、四实验臂共享候选、随机对照复现性、指标和配对统计。
