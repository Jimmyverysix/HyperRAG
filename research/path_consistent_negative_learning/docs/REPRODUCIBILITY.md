# Retriever-only 正式实验复现说明

## 1. 环境与边界

- 研究分支：`codex/www-retriever-only`
- 官方上游提交：`6d5a9033353c516a9220d78591f2c666f19ee0b1`
- 服务器 Python：`/root/miniconda3/envs/hyperrag_official/bin/python`
- GPU：只允许使用 0--5 号 RTX 3090，最多六个独立进程，不使用 DDP
- 正式配置：`configs/retriever_only/wiki_main.json`
- 服务器运行根目录：`/root/hyperrag_pcneg/runs/retriever_only`

本地与服务器的代码只通过 GitHub 快进同步。数据、GTE 张量、checkpoint 与 raw log 不进入 Git；聚合 JSON/CSV、论文表图和最终 `main.pdf` 才回到版本库。不要停止 GPU 0 上不属于本研究的历史进程。

整个正式流程不使用生成式 LLM，不调用 OpenAI、百炼或其他付费 API。唯一文本模型是本地 `Alibaba-NLP/gte-large-en-v1.5`；它只产生向量。topic entity 来自结构化 query，答案只用于训练监督和评价。

## 2. 数据与固定变量

```bash
PY=/root/miniconda3/envs/hyperrag_official/bin/python
REPO=/root/hyperrag_pcneg/work/HyperRAG_official
STRUCTURED=/root/hyperrag_pcneg/data_sources/WikiTopics_QE
NLG=/root/hyperrag_pcneg/data_sources/WikiTopicsQE_NLG
MODEL=/root/hyperrag_pcneg/data_sources/models/gte-large-en-v1.5
RUN=/root/hyperrag_pcneg/runs/retriever_only
CONFIG=research/path_consistent_negative_learning/configs/retriever_only/wiki_main.json
LABELS=research/path_consistent_negative_learning/artifacts/retriever_only/raw/label_snapshot.json
DOMAINS="art award edu health infra loc org people sci sport tax"
cd "$REPO"
export PYTHONPATH="$REPO"
```

图实体以 Wikidata QID 保持身份；冻结英文标签只作为 GTE 文本。同名实体不能合并。正式训练种子为 42--46，lambda 网格为 `0.00, 0.10, 0.25, 0.50, 0.75, 1.00`。

正式 `prepare` manifest 只能在 art/valid 的 beam=10 与 beam=32 P0 完成后生成。唯一选择指标是候选路径覆盖率；覆盖率较高者胜，精确并列取 10。实测 beam=10 为 13.10%（1,294/9,878），beam=32 为 21.92%（2,165/9,878），因此正式宽度冻结为 32；完整机器可读记录见 `artifacts/retriever_only/p0/beam_selection.json`，不追加第三个宽度。

## 3. 预检与测试

```bash
$PY -m unittest discover \
  -s research/path_consistent_negative_learning/tests -p 'test_*.py'

$PY -m research.path_consistent_negative_learning.retriever_only.validate \
  --structured-root "$STRUCTURED" --nlg-root "$NLG" \
  --label-snapshot "$LABELS" --domains $DOMAINS \
  --output "$RUN/preflight/data_validation.json"
```

预检必须在任何正式 GPU sweep 前通过，并报告每领域重复英文标签数量、结构/NLG 对齐数和可评价问题数。

## 4. 编码、候选准备与验证集扫参

下列三个 phase 分别生成 manifest；GPU 队列会跳过已经完整产生预期文件的作业。

```bash
for PHASE in encode prepare sweep; do
  $PY -m research.path_consistent_negative_learning.scripts.build_retriever_manifest \
    --phase "$PHASE" --config "$CONFIG" \
    --structured-root "$STRUCTURED" --nlg-root "$NLG" \
    --label-snapshot "$LABELS" --model-path "$MODEL" \
    --run-root "$RUN" --output "$RUN/manifests/$PHASE.json"
  $PY -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
    --manifest "$RUN/manifests/$PHASE.json" --gpus 1 2 3 4 5
done
```

每个领域只用 validation Answer-Path MRR 选择 lambda；完全并列时取更大的 lambda。选择文件产生前，`prepare-test`、`main-test` 与两个 `path-sensitivity-*` manifest 都会拒绝生成。

```bash
SELECTION="$RUN/selection/lambdas.json"
$PY -m research.path_consistent_negative_learning.scripts.select_retriever_lambdas \
  --run-root "$RUN" --domains $DOMAINS --output "$SELECTION"
```

## 5. 冻结测试、匹配随机与路径择一敏感性

先运行冻结测试。路径择一敏感性使用两个 phase：`path-sensitivity-prepare` 对同一“领域 × 路径变体”的五个训练种子共享与种子无关的图计算，`path-sensitivity-train` 再执行与原协议完全相同的 330 个训练和评估作业。旧的 `path-sensitivity` 一体式 phase 仅为历史复现保留，不用于新的正式运行。

```bash
for PHASE in prepare-test main-test; do
  $PY -m research.path_consistent_negative_learning.scripts.build_retriever_manifest \
    --phase "$PHASE" --config "$CONFIG" \
    --structured-root "$STRUCTURED" --nlg-root "$NLG" \
    --label-snapshot "$LABELS" --model-path "$MODEL" \
    --run-root "$RUN" --selection-file "$SELECTION" \
    --output "$RUN/manifests/$PHASE.json"
  $PY -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
    --manifest "$RUN/manifests/$PHASE.json" --gpus 1 2 3 4 5
done

for PHASE in path-sensitivity-prepare path-sensitivity-train; do
  $PY -m research.path_consistent_negative_learning.scripts.build_retriever_manifest \
    --phase "$PHASE" --config "$CONFIG" \
    --structured-root "$STRUCTURED" --nlg-root "$NLG" \
    --label-snapshot "$LABELS" --model-path "$MODEL" \
    --run-root "$RUN" --selection-file "$SELECTION" \
    --output "$RUN/manifests/$PHASE.json"
done

# 候选准备受单核图遍历限制，DDE 显存很小；用同卡多槽填满空闲 CPU。
$PY -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
  --manifest "$RUN/manifests/path-sensitivity-prepare.json" \
  --gpus 0 1 2 3 4 5 --workers-per-gpu 6

# 训练按实测峰值显存拆分。award/health 保持每卡一个作业。
$PY -m research.path_consistent_negative_learning.scripts.build_retriever_manifest \
  --phase path-sensitivity-train --config "$CONFIG" \
  --structured-root "$STRUCTURED" --nlg-root "$NLG" \
  --label-snapshot "$LABELS" --model-path "$MODEL" \
  --run-root "$RUN" --selection-file "$SELECTION" \
  --domains award health \
  --output "$RUN/manifests/path-sensitivity-train-heavy.json"
$PY -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
  --manifest "$RUN/manifests/path-sensitivity-train-heavy.json" \
  --gpus 1 2 3 --workers-per-gpu 1

# art 与其余八个较小领域每卡三个独立作业；只改变调度，不改变作业内计算。
$PY -m research.path_consistent_negative_learning.scripts.build_retriever_manifest \
  --phase path-sensitivity-train --config "$CONFIG" \
  --structured-root "$STRUCTURED" --nlg-root "$NLG" \
  --label-snapshot "$LABELS" --model-path "$MODEL" \
  --run-root "$RUN" --selection-file "$SELECTION" \
  --domains art edu infra loc org people sci sport tax \
  --output "$RUN/manifests/path-sensitivity-train-light.json"
$PY -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
  --manifest "$RUN/manifests/path-sensitivity-train-light.json" \
  --gpus 1 2 3 --workers-per-gpu 3
```

主实验名称固定为策略1（Baseline）、策略2（Matched Random）和策略3（Ours）。策略2逐题从全部已采样负例中均匀无放回抽取 `|D_q|` 个，不查看路径身份；允许偶然与 `D_q` 重合。敏感性统计只聚合至少一个 topic--answer 对具有多条等长最短路径的测试问题；先在每个路径变体内平均五个训练种子，再对三个变体均值计算 mean、standard deviation 和 range。

批量预处理保持每个种子独立的 `random.Random(seed)` 流，候选、标签、路径一致掩码及其顺序与逐种子运行相同。`--domains` 只从冻结配置中选取一个无重复的有序子集，两个训练清单的并集仍是原始 330 个作业；`workers-per-gpu` 只改变独立作业的调度并发，不改变任何作业内部执行顺序。DDE 仍逐题调用原实现；CUDA `float32` 归约允许出现至多一个 ULP 的数值噪声，不改变协议、候选或监督信号。若中途恢复，完整的 `training.pt` 与对应 report 会被跳过；只存在其中一个文件时会拒绝覆盖并明确报错。

## 6. 聚合与论文产物

```bash
mkdir -p "$RUN/aggregates"
for PHASE in prevalence main-test sensitivity; do
  $PY -m research.path_consistent_negative_learning.scripts.aggregate_retriever_results \
    --phase "$PHASE" --run-root "$RUN" --domains $DOMAINS \
    --output "$RUN/aggregates/$PHASE.json" \
    --csv-output "$RUN/aggregates/$PHASE.csv"
done

$PY -m research.path_consistent_negative_learning.scripts.extract_retriever_case_study \
  --structured-root "$STRUCTURED" --nlg-root "$NLG" \
  --label-snapshot "$LABELS" --domain art --seed 42 \
  --output "$RUN/aggregates/case_study.json"
```

把小型 aggregate 产物同步到 `artifacts/retriever_only/` 后，分别运行：

```bash
$PY -m research.path_consistent_negative_learning.scripts.generate_retriever_paper_results --help
$PY -m research.path_consistent_negative_learning.figures.gen_retriever_only --help
```

论文唯一主文件是 `paper/www2027/main.tex`，必须在该目录编译为 `main.pdf`。结果链固定为：

```text
raw artifacts -> aggregate JSON/CSV -> LaTeX macros/tables -> PDF/SVG figures -> main.pdf
```

## 7. 恢复与失败定位

每个作业保存完整 command、stdout/stderr、Git commit、dirty state、环境、GPU、seed、lambda、逐题指标与 checkpoint 指针。若 CUDA OOM，先确认是否一卡一进程；只有编码阶段允许统一降低 GTE batch size，因为它只改变吞吐，不改变输出定义。训练 batch size 属于冻结协议，不得按方法单独修改。

队列只在全部作业成功后写 `.complete`；失败清单写入 `.failures.json`。恢复时重新运行同一 manifest，不删除历史 runs，不覆盖元数据口径不同的目录。敏感性预处理结束后必须先确认 `path-sensitivity-prepare.complete`，再启动训练 phase；这样不会让预处理与训练争抢同一张卡。
