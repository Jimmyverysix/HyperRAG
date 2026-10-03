# 最终结果与交付状态

状态日期：2026-10-04。全部必需主实验和已实现的两组全最短路对照完成；没有本轮 GPU 训练仍在运行。

## 真实完成的实验

| 数据 | 已复用模型 | 新完成模型 | 策略与覆盖 |
|---|---:|---:|---|
| WikiTopics 十一领域 | 165 | 55 | 旧三策略 + 新 Positive Relabel，各 42–46 |
| MetaQA-3hop | 15 | 5 | 旧三策略 + 新 Positive Relabel，各 42–46 |
| PathQuestion | 0 | 25 | 四策略 × 五种子，以及 All-shortest × 五种子 |
| KQAPro-entity | 0 | 25 | 四策略 × 五种子，以及 All-shortest × 五种子 |

合计新增 **110** 次完整训练/评价、复用 180 次，主实验 280 个模型结果，另有 10 个构造对照。新增每次均保存 checkpoint、config、training history 和逐题 metrics；本地 `new_run_registry.json` 保存真实路径、配置、时间、commit、训练历史及可用的 GPU 调度记录。完整大张量、checkpoint 和逐题原始报告保留在服务器 `/root/hyperrag_pcneg/runs/cross_dataset_20261003/`；旧目录不覆盖。

主训练阶段使用物理 GPU 0、2、3、4、5、6 六张 RTX 3090，同卡最多三进程，实测利用率接近 99%。不使用第七张卡，不改变 batch、FP32 或最大 epoch 来加速。编码用六个 shard；缓存 topic BFS 和索引读取减少 CPU/内存瓶颈。全最短路的已运行清单是两条短链；最终复现清单拆成两项准备和十项可并发种子，消除后续串行等待。

## 四策略主指标

下表 APC-MRR 为百分数。WikiTopics 领域等权，其余为选定 evaluation split 全题平均。

| 数据 | Original | Random Mask | PCN Mask | Positive Relabel | Candidate Oracle |
|---|---:|---:|---:|---:|---:|
| WikiTopics（领域等权） | 6.054 | 6.077 | 6.312 | 6.312 | 32.734 |
| MetaQA-3hop | 20.862 | 20.658 | 22.185 | 19.393 | 56.018 |
| PathQuestion | 42.917 | 43.647 | 45.097 | 44.802 | 93.952 |
| KQAPro-entity | 19.981 | 20.372 | 21.113 | 21.336 | 82.692 |

配对 APC-MRR 差值和 95% CI，单位为百分点：

| 数据 | Mask−Original | Mask−Random | Mask−Relabel |
|---|---:|---:|---:|
| WikiTopics（领域等权） | 0.257 [0.224, 0.291] | 0.235 [0.200, 0.270] | -0.000 [-0.027, 0.027] |
| MetaQA-3hop | 1.323 [0.825, 1.825] | 1.528 [1.022, 2.037] | 2.792 [2.525, 3.065] |
| PathQuestion | 2.179 [1.408, 3.004] | 1.449 [0.912, 1.996] | 0.295 [-0.302, 0.875] |
| KQAPro-entity | 1.132 [0.638, 1.645] | 0.741 [0.346, 1.145] | -0.223 [-0.454, -0.000] |

每题先平均五个配对种子，再做 10,000 次题级有放回 bootstrap；WikiTopics 按领域重采样后等权平均。区间条件于这五次训练，未作多重比较校正。KQAPro 的 Mask−Relabel 上界原值约 −0.0000699 个百分点，显示为 −0.000 不代表稳健的普适优势。MetaQA 重标记五种子波动较大，不能用题级 CI 代替训练随机性评价。

Reach@5、Reach@10、全部题/可达题两套结果、逐种子均值、全部比较 CI 及每一领域都在 `summary.json` / `main_results.csv`。MetaQA 旧 CSV 缺 Reach@5 列，已从 RR≥1/5 精确恢复，并与已有每种子 JSON 一致；没有填零或估计。

## 全最短路与主张边界

| 数据 | All-shortest APC-MRR (%) | All-shortest−Mask [95% CI]（百分点） |
|---|---:|---:|
| PathQuestion | 46.714 | 1.617 [0.191, 3.072] |
| KQAPro-entity | 31.779 | 10.666 [7.064, 14.231] |

两组全最短路构造都高于屏蔽。它改变正例并重采样负例，区别于只给实际 PCN 改标的 Positive Relabel。PathQuestion 正例从 11,728 增至 17,024，KQAPro 从 7,562 增至 46,184；同一训练题集合，seed 42 的新负例分别为 15,975、46,141，PCN 均为零。真实计数与来源见 `all_shortest_training_counts.json`。

因此论文的成立结论是：固定原监督集合时，精确撤销 PCN 负监督改善四组数据的 APC-MRR，且优于等量随机撤销；不能写屏蔽是最佳多路径训练方案，也不能写统一优于重标记。WikiTopics award 领域 Mask−Original 为 −0.172 个百分点，95% CI [−0.274, −0.068]，新稿保留此失败。

## 现象与机制记录

四组实际 PCN/负例分别为 4.588%、11.808%、8.796%、1.445%。`prevalence_summary.json` 为题数与五种子样本总数；`prevalence_per_seed.csv` 和 `prevalence_count_distribution.csv` 为逐种子比例、受影响均值/中位数及完整计数分布。WikiTopics 保留领域及题级汇总两种单位。

新基准机制 JSON 包含路径多重性、替代路径数、跳数、关系类型、候选数、Oracle、实际训练 topic PCN 比例/每题个数，以及训练 PCN 最终 rank。密度匹配仅覆盖 PathQuestion 691/711（比例有定义 690）及 KQAPro 190/468；没有把未匹配题当零。KQAPro 高密度小桶不足以支持趋势结论。分桶及训练 rank 是描述性证据，不是独立因果机制证明。

## 失败、恢复和未运行范围

| 项目 | 最终状态 | 原因与处理 |
|---|---|---|
| 六卡主实验 | 完成 | 三进程/卡的大型物化张量曾 OOM；原日志保留，20 项未完成/失败任务在 recovery_indexed 独立目录用相同 FP32 索引读取完成 |
| HF / GitHub 服务端下载 | 已绕过 | 超时；用本地下载原文件后 SFTP 中转，代码通过 Git bundle 同步 |
| WebQSP | 本轮未接入 | 已审查官方问题；现成关系裁剪图读取问题 parse，需独立 Freebase 快照后才能接入 |
| GrailQA | 本轮未接入 | 下载入口返回网页，完整数据及独立 Freebase/Virtuoso 服务未就绪；没有伪造统计 |
| WikiTopics / MetaQA All-shortest | 未运行 | 可选构造仅实现并完成两个新普通 KG；没有声称覆盖既有超图或旧训练协议 |
| Codex 内置 LaTeX 编译器 | 环境不可用 | 报 Unable to find standard directories for platform；已用既有 XeLaTeX 编译最终 PDF，不安装新运行时 |

本轮必需证据没有处于 blocked 或正在运行状态。WebQSP/GrailQA 是明确暂缓的候选，不是已完成基准。PathQuestion 为模板数据、自定义划分；KQAPro 为实体单 topic 子集，evaluation 使用官方 val；WikiTopics 使用既有对齐后可评价分母和传导式图。作者信息仍待填写，论文没有未完成实验数字占位。

## 新稿、润色与复现

根目录 `paper_restructured_zh.tex` / `paper_restructured_zh.pdf` 是完整新中文稿，独立于旧稿。十个主章节、形式命题与证明、六个 RQ、四策略、全最短路对照、条件指标、机制、局限及参考文献均已写入。采用 nature-writing 起稿、nature-polishing 独立润色，保留初稿和具体修改记录；用 nature-academic-search 核验实际引用，用 nature-figure 导出科学图。

最终 PDF 12 页；XeLaTeX 成功，图/表/中文与公式已渲染检查。图附可编辑 SVG、PDF、300 dpi PNG、PGF 和 CSV。入口与命令见 `RUNBOOK_ZH.md`，改动见 `IMPLEMENTATION_CHANGELOG_ZH.md`。开发分支 `codex/pcn-cross-dataset` 保留阶段提交；用户既有改动没有混入本轮提交。
