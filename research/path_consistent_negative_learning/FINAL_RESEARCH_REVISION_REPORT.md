# WWW 2027 研究修订最终报告

日期：2026-09-20  
工作分支：`www-path-supervision-revision`  
本轮已推送提交：`ad7477a`、`3f1581b`、`58519d4`、`45c42dc`  
远端分支：`github/www-path-supervision-revision`

## 结论先行

本轮已经完成研究工程、冻结协议、核心实现、控制实验编排、统计聚合、图表生成、中文论文框架和结果自动同步链路；81 项测试全部通过，历史产物的程序化交叉核验也全部通过。

但本轮尚未产生 dataset-specific λ 的正式 GPU 主结果，因此不能宣称这项研究已经最终成功。服务器在本轮最后一次复核时仍无法访问 GitHub，45 秒后以退出码 124 超时；服务器代码停留在 `31db6f0`，无法按既定同步约束取得当前分支。官方 HyperRAG Retriever、端到端 QA 和开放域实验还缺少完整 NLG、构图后 GraphML、GTE 表示、Retriever checkpoint、端到端输出及 API 配置。语义审计还需要两名真人独立标注。

当前最准确的科研判断是：历史实验为“降低路径一致负例的负监督强度”提供了有希望的机制证据，但新的冻结协议尚未完成实证闭环。历史 λ=0 失败结论保持不变；是否能通过 dataset-specific λ 把排序损失稳定控制在 1 个百分点以内，必须由尚未运行的 selection/test 实验回答。

## A. 仓库修改摘要

1. 保留所有历史 audit、Gate C、加权确认、跨领域 transfer、proposal 和 PDF；旧产物未覆盖，新产物限定写入 `artifacts/www_revision/`。
2. 将完整主题—答案最短路径条件、加权 BCE 和 λ 选择拆成独立模块，未增加网络模块、可学习参数或推理期开销。
3. 冻结 11 个 WikiTopics_QE 领域、60/10/15/15 问题划分、λ 网格 `{0, 0.1, 0.25, 0.5, 0.75, 1}`、五个共享种子 42–46 和统一选择规则。
4. 实现 baseline、λ=0 端点、dataset-specific λ、历史 λ=0.1 transfer、逐题等量 matched-random 和 all-positive diagnostic 的编排与聚合。
5. 每个运行保存 Git 提交、dirty state、完整配置、数据版本、命令、种子、λ、GPU、Python/PyTorch/CUDA/驱动版本和起止时间；已有结果缺少 provenance 时拒绝复用。
6. 实现盲化双人语义审计的抽样与分析流程；程序不自动生成共识标签，也不把模型标签冒充人工标签。
7. 重写 WWW 2027 中文主文档，并将方法、case study、实验问题、讨论、限制和结果表统一放在同一份主文档中。旧 `proposal/` 只作为历史 provenance 保留，不再与另一份“实验报告”并行维护。
8. 论文数字只能由聚合 JSON/CSV 自动生成；缺少真实结果时编译主动失败，不允许占位数字进入 PDF。

## B. 新增与修改文件

核心实现：

- `path_supervision/path_consistency.py`
- `path_supervision/weighted_loss.py`
- `path_supervision/lambda_selection.py`
- `scripts/prepare_www_proxy_data.py`
- `scripts/build_www_job_manifest.py`
- `scripts/run_experiment_queue.py`
- `scripts/evaluate_checkpoint.py`
- `scripts/select_lambda.py`
- `scripts/select_all_lambdas.py`
- `scripts/aggregate_audit_distribution.py`
- `scripts/aggregate_www_proxy_results.py`
- `scripts/prepare_semantic_audit.py`
- `scripts/analyze_semantic_audit.py`
- `scripts/generate_paper_results.py`
- `scripts/verify_historical_artifacts.py`

冻结配置、协议和文档：

- `configs/www_revision/proxy_main.json`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/METHOD_SPEC.md`
- `docs/REPRODUCIBILITY.md`
- `docs/SEMANTIC_AUDIT_GUIDE.md`
- `docs/CITATION_AUDIT.md`
- `REPO_AUDIT.md`
- `IMPLEMENTATION_PLAN.md`
- `CHANGELOG_WWW_REVISION.md`
- `Makefile`
- `README.md`

论文与图表：

- `paper/www2027/main.tex`
- `paper/www2027/references.bib`
- `figures/www_style.py`
- `figures/gen_www_method.py`
- `figures/gen_www_conflict.py`
- `figures/gen_www_lambda.py`
- `figures/gen_www_tradeoff.py`
- `artifacts/www_revision/figures/method_overview.pdf`
- `artifacts/www_revision/figures/method_overview.svg`

测试集中新增了正式路径定义、加权端点、λ 选择、历史核验、作业清单、聚合、语义审计和论文生成测试。完整清单见 `tests/`。

## C. 最终数学定义

知识超图为 $H=(V,F)$，实际距离在关联图 $G_I=(V\cup F,E_I)$ 上计算。对问题 $q$，主题实体集合为 $S_q$，答案实体集合为 $A_q$，官方流程实际采样的负例集合为 $N_q$。

候选转移 $t=(v,f,u)$ 属于路径一致负例集合 $D_q\subseteq N_q$，当且仅当存在 $s\in S_q$ 和 $a\in A_q$，使

$$
d_I(s,v)+2+d_I(u,a)=d_I(s,a).
$$

其中 `+2` 对应 $v\rightarrow f\rightarrow u$ 的两条关联边。实现使用源距离与答案反向距离，不枚举所有最短路径。局部降距条件不作为主定义。

样本权重为

$$
w_x=\begin{cases}
\lambda,&x\in D_q,\\
1,&\text{其他样本},
\end{cases}
\qquad 0\leq\lambda\leq1,
$$

训练目标为

$$
\mathcal L_\lambda=
\frac{\sum_x w_x\,\operatorname{BCE}(x)}{\sum_x w_x}.
$$

标签不翻转。λ=1 精确恢复 baseline，λ=0 精确恢复正确归一化的完全屏蔽端点，中间值表示弱负监督。

## D. λ 的正式选择规则

- 每个领域只使用自己的 selection split 选择一个 $\lambda_D^*$。
- 固定网格为 `{0.00, 0.10, 0.25, 0.50, 0.75, 1.00}`，不得看 test 后插值或扩展。
- 当前结构代理协议统一使用 selection answer reach@10；不构造 answer reach 与 PR-AUC 的人工复合分数。
- 分数并列时按冻结规则选择更大的 λ，即使用更接近 baseline 的更保守权重。
- λ 冻结后才生成 test 作业清单；选择器和清单构造测试会阻止 test 泄漏。
- 历史 λ=0.1 只作为固定权重跨领域 transfer 证据，不代替新的 tuned main result。

## E. 已完成实验与核验

### 已完成

1. 历史 artifact 程序化交叉核验：通过，输出 `artifacts/www_revision/provenance/historical_verification.json`。
2. λ=1 与 baseline loss 等价、λ=0 与归一化 mask loss 等价：通过单元测试。
3. 两条等长最短路径、非最短绕路和“局部降距但完整路径不一致”反例：通过单元测试。
4. selection/test 隔离、并列规则、逐题 matched-random 数量匹配、运行 provenance、聚合配对和论文生成守卫：通过单元测试。
5. WWW 方法示意图：已生成 PDF 和 SVG，并完成视觉检查。
6. 中文 ACM 主文档：已用合成输入完成 XeLaTeX 冒烟编译；合成数字、合成图和冒烟 PDF 已全部删除。

### 尚未完成

- 本轮完整 11 领域 audit 重新聚合。
- 11 领域 × 5 seeds × 6 λ 的正式 selection sweep。
- dataset-specific λ 冻结后的 test checkpoint 与 matched-random/all-positive 对照。
- 正式结果聚合、置信区间和三张结果图。
- official HyperRAG Retriever、端到端 QA、开放域 benchmark。
- 双人语义标注及 Cohen's kappa/共识分析。

## F. 精确复现命令

服务器从 GitHub 恢复同步后，在仓库根目录执行：

```bash
git fetch https://github.com/Jimmyverysix/HyperRAG.git www-path-supervision-revision
git switch -C www-path-supervision-revision FETCH_HEAD

make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python test
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python verify-history
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python \
  AUDIT_RAW=/root/hyperrag_pcneg/runs/wikitopics_audit audit
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python \
  DATA_SOURCE=/root/hyperrag_pcneg/data_sources/WikiTopics_QE prepare
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python sweep-manifest
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python \
  GPU_IDS="0 1 2 3 4 5" sweep
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python select-lambda
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python test-manifests
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python \
  GPU_IDS="0 1 2 3 4 5" main-experiments
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python aggregate
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python figures
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python paper-results
make -C research/path_consistent_negative_learning \
  PYTHON=/root/miniconda3/envs/sdhp/bin/python paper
```

作业队列可恢复：只有 provenance 与冻结配置完全匹配的完成结果才跳过；不完整或不匹配的目录会报错，不会静默覆盖。

## G–I. 主要结果及其来源

下列全部是历史结果，经本轮程序重新核验，不是本轮新 GPU 结果：

| 事实 | 核验值 |
|---|---:|
| 领域数 | 11 |
| 问题数 | 90,000 |
| 组合候选数 | 872,576,411,856 |
| 路径一致候选数 | 42,926,111 |
| 候选空间比例 | 0.004919% |
| 实际采样负例数 | 10,156,714 |
| 其中路径一致负例数 | 586,117 |
| 实际采样比例 | 5.770735% |
| 领域范围 | 1.187911%–11.674364% |
| λ=0 的 selected-path PR-AUC 变化 | −1.318725 个百分点 |
| 历史 λ=0 Gate C | 不扩展完整训练 |
| 固定 λ=0.1 transfer 的宏平均 answer reach@10 变化 | +4.806555 个百分点 |
| 固定 λ=0.1 transfer 的宏平均 selected-path PR-AUC 变化 | −0.217250 个百分点 |
| answer reach 改善领域数 | 10/11 |

历史 λ=0.1 的 answer reach 变化 95% 区间为 `[+1.597693, +9.393634]` 个百分点，selected-path PR-AUC 变化 95% 区间为 `[−0.340475, −0.102108]` 个百分点。

本轮新性能结果：无。新的 `historical_verification.json` 是核验产物，不是一次新模型实验。因而不得把上表写成 dataset-specific λ 主结果，也不得据此宣称 official HyperRAG QA 获得提升。

## J–L. Pending、端到端和语义审计状态

1. **dataset-specific 主实验：pending。** 原因是服务器无法通过 GitHub 取得当前代码。
2. **official HyperRAG end-to-end：未完成。** 当前机器缺少正式 Retriever/QA 所需资产和 API 配置；结构代理不会冒充端到端结果。
3. **开放域 benchmark：未完成。** 需要先取得并核验官方数据与模型资产。
4. **semantic human audit：流程已完成，真人标注未完成。** 需要两名真实标注者独立填写盲化表，再由脚本计算一致率和 kappa；程序不会替代人工共识。

## M. Figure 路径

已存在：

- `artifacts/www_revision/figures/method_overview.pdf`
- `artifacts/www_revision/figures/method_overview.svg`

真实实验完成后自动生成，目前不存在：

- `artifacts/www_revision/figures/path_conflict.pdf` 与 `.svg`
- `artifacts/www_revision/figures/coverage_ranking_tradeoff.pdf` 与 `.svg`
- `artifacts/www_revision/figures/lambda_sensitivity.pdf` 与 `.svg`

## N. Table 路径

以下表格由真实聚合结果自动生成，目前不存在：

- `paper/www2027/generated/tables/main_proxy.tex`
- `paper/www2027/generated/tables/audit.tex`
- `paper/www2027/generated/tables/lambda.tex`

## O. 论文 PDF 状态

统一主文档是 `paper/www2027/main.tex`。它包含研究动机、正式定义、最小 case study、完整实验协议、RQ1–RQ4 结果位置、讨论、限制、结论和附录，不再另设一份活跃“实验报告”。旧 `proposal/main.tex` 与用户修改过的 `proposal/main.pdf` 作为历史记录保留且未覆盖。

最终 PDF 约定路径是 `paper/www2027/main.pdf`。当前该文件有意不存在：真实聚合结果尚未生成，留下冒烟编译 PDF 会把合成内容误认为正式结果。待 `make paper-results` 成功后，`make paper` 才会生成正式 `main.pdf`。

## P. 测试结果

- Python 单元测试：81/81 通过。
- `git diff --check`：通过。
- 历史 artifact 交叉校验：全部 15 项检查通过。
- XeLaTeX 合成输入冒烟编译：通过，5 页；合成输入与输出已删除。
- 当前唯一未暂存修改：用户原有的 `proposal/main.pdf`，本轮未改写、未暂存。

## Q. Blocker

1. **服务器到 GitHub 的网络阻塞。** 2026-09-20 最后复核：服务器 HEAD 为 `31db6f0`；访问 GitHub 分支 45 秒超时，退出码 124。按用户规定，服务器与本地只能通过 GitHub 同步，因此没有使用 SCP、压缩包或第三方镜像绕过。
2. **官方端到端资产缺失。** 缺少完整 NLG、GraphML、GTE embeddings、Retriever checkpoint、端到端结果与 API 配置。
3. **真人标注资源缺失。** 语义审计不能由代码或 LLM 伪装完成人工判断。

## R. 最值得人工完成的三件事

1. 恢复服务器到 `github.com`/`api.github.com` 的 HTTPS 访问，或由管理员配置允许的 GitHub 网络出口；随后按 F 节命令启动六卡队列。
2. 提供或确认 HyperRAG 官方 Retriever/QA 所需的完整数据、checkpoint 和 API 配置，并记录来源与版本，才能开展 official end-to-end 和开放域评测。
3. 安排两名真实标注者完成至少每领域 20 个候选的盲化语义审计，再运行 `analyze_semantic_audit.py` 生成一致性与共识结果。

## 最终科研判断

历史证据支持继续研究：路径一致负例在实际采样负例中并不罕见；完全屏蔽会提高覆盖但造成超过预设保护线的排序损失；较弱的 λ=0.1 历史 transfer 在 10/11 个领域提高 answer reach，并把宏平均排序下降缩小到约 0.22 个百分点。

然而，这只能说明“加权改进值得进入正式验证”，不能替代新的 dataset-specific selection、matched-random/all-positive 控制、official retrieval/QA 和人工语义审计。因此当前结论是：**研究工程升级成功，历史机制证据积极，但新正式研究结论尚未闭环，暂不能宣称整项研究已经成功。**
