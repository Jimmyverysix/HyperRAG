# WWW 修订版复现说明

## 1. 环境与资源

- 代码分支：`www-path-supervision-revision`
- 服务器 Python：`/root/miniconda3/envs/sdhp/bin/python`
- GPU：仅允许使用 RTX 3090 的 0--5 号卡；每个训练任务独占一张卡，不使用 DDP。
- 正式配置：`configs/www_revision/proxy_main.json`
- 新产物根目录：`artifacts/www_revision/`

服务器代码只能通过 GitHub 同步。本轮检查时服务器到 GitHub HTTPS 和 `ssh.github.com:443` 均超时；网络恢复前不得用复制源码的方式绕过。

## 2. 数据

结构代理实验使用完整 WikiTopics_QE 11 领域整数图。服务器路径为 `/root/hyperrag_pcneg/data_sources/WikiTopics_QE`。历史逐题审计位于 `/root/hyperrag_pcneg/runs/wikitopics_audit`。

仓库和服务器均缺少官方完整 NLG、构建后 GraphML、GTE 表示、Retriever checkpoint、端到端输出与 API 配置；因此当前命令不会产生 official HyperRAG Retriever/QA 结果。

## 3. 最小复现顺序

在 `research/path_consistent_negative_learning/` 下执行：

```bash
make test
make verify-history
make audit AUDIT_RAW=/root/hyperrag_pcneg/runs/wikitopics_audit
make prepare DATA_SOURCE=/root/hyperrag_pcneg/data_sources/WikiTopics_QE
make sweep-manifest
make sweep
make select-lambda
make test-manifests
make main-experiments
make aggregate
make figures
make paper-results
make paper
```

`make sweep` 先执行 11×5×6 个 selection 任务。`make select-lambda` 为每个领域写出独立 `lambda_star.json`。`make test-manifests` 只有在所有 λ 文件存在后才生成测试任务；`make main-experiments` 先评估冻结检查点，再训练 matched-random 与 all-positive 对照。

## 4. 恢复与失败定位

GPU 队列在每个实验目录保存 `command.json`、`stdout.log`、`stderr.log`、`result.json`、逐题指标与 checkpoint。重跑时，元数据完全匹配的结果自动跳过；不匹配则失败，不覆盖。队列根目录只有全部任务成功后才写 `COMPLETE`。

若出现 CUDA OOM，先检查任务是否确实一卡一进程，再减小 batch size；baseline 与所有实验臂必须同步修改并生成新配置，不得只改某个方法。若某领域输入缺失，数据准备阶段会在启动 GPU 前失败。

## 5. 随机性与统计

采样/训练种子固定为 42--46；问题划分种子为 20260920。lambda selection 只读取 selection split。正式差异按 query key 配对，先跨 seed 等权平均，再进行 10,000 次 paired bootstrap；领域宏平均以领域为等权单位。

## 6. 论文同步

表格、宏与图片必须由聚合 JSON/CSV 自动生成。禁止在 TeX 中手填实验数值。论文生成后应核对：源 JSON、生成 TeX、图注、正文结论四者口径一致。
