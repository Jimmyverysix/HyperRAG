# WWW 修订版正式实验协议

状态：在新主实验结果产生前冻结。配置真源为 `configs/www_revision/proxy_main.json`。

## 1. 证据范围

本协议的 P0 对象是官方 WikiTopics 整数图、官方候选构造语义与当前两层结构 Retriever 的受控结构代理实验。它不是原版 HyperRAG Retriever 或端到端 QA。服务器缺少官方 GraphML、GTE 表示、Retriever 检查点、完整 NLG 与 API 配置，因此任何 P0 数值只能称为结构代理证据。

11 个 WikiTopics 领域分别作为独立 dataset：art、award、edu、health、infra、loc、org、people、sci、sport、tax。

## 2. 数据与划分

- 每个 sampler/training seed 独立复现官方等量负例采样，种子为 42、43、44、45、46。
- 所有方法共享完全相同的逐题候选、特征和问题级划分。
- 固定 split seed 为 20260920，比例为 60% train、10% validation、15% selection、15% test。
- `validation` 只用于训练早停；`selection` 是独立超参数验证集；`test` 在每个 dataset 的 λ 冻结前禁止访问。

## 3. λ 选择

固定搜索空间为 `{0, 0.1, 0.25, 0.5, 0.75, 1}`，不追加事后网格点。每个 dataset 独立最大化 selection split 的 `answer_reachability@10`，跨五个种子等权平均。它是单一结构代理指标，不与 PR-AUC 组成复合分数。若均值在绝对误差 `1e-12` 内并列，选择更大的 λ，即更接近原监督、干预更弱的值。

选择器只接受 `evaluation_split=selection` 的完整六点 sweep，并输出 `lambda_star.json`。测试评估必须读取该文件，不接受命令行手填 λ。

选择 `answer_reachability@10` 的原因是服务器当前无法计算官方 QA F1/EM 或官方文本 Retriever 指标，而 validation weighted BCE 在不同 λ 下改变了监督权重，横向比较并不公平。本指标仍是代理指标，不能写成 QA 准确率。

## 4. 实验臂

- baseline：λ=1，与原二元监督数值等价；
- mask endpoint：λ=0，与删除路径一致负例后的按有效权重归一化损失等价；
- tuned：各 dataset 冻结的 λ*；
- matched random：逐题随机选择 `|D_q|` 个普通负例，赋予与 tuned 相同的 λ；
- all-shortest-positive：把路径一致候选翻为正例，仅作诊断；
- fixed λ=0.1：只使用历史独立确认与跨领域结果，作为零重调迁移证据，不替代 tuned 主结果。

除标签/权重操作外，各臂共享候选、数据划分、模型、优化器、早停和训练预算。Matched random 使用共享 seed，并逐题匹配降权数量。

## 5. 指标与统计

P0 primary metric 是 `answer_reachability@10`。secondary metrics 是 selected-path MRR、selected-path PR-AUC、all-shortest MRR 与 all-shortest PR-AUC。报告五个种子的 mean±std；方法比较按 query key 配对，先在同一问题上跨种子等权平均，再做 10,000 次配对 bootstrap，随机种子 20260920，报告 percentage-point difference 与 95% CI。

领域宏平均以 dataset 为等权单位，同时保留逐领域结果。λ* 与 sampled conflict rate 的 Spearman 相关仅作探索性分析，不作因果结论。

## 6. 正式端到端边界

official HyperRAG Retriever/QA 只有在取得完整 NLG、构建后 GraphML、GTE 表示、官方或可重训检查点以及所需 API 配置后才能运行。届时应另建协议，优先按官方 validation 指标选 λ；不得把本协议的代理结果改名为端到端结果。
