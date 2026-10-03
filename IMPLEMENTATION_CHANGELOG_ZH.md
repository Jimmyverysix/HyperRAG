# 实现变更记录

开发分支 `codex/pcn-cross-dataset`。旧论文、旧结果和用户工作区已有改动保留。本轮先提交审计，再分阶段实现；历史提交和训练目录的旧名称只作为来源标识，不用于新论文叙事。

1. `07129f4`：项目审计，定位真实 WikiTopics/MetaQA 结果及监督链路。
2. `843e0d2`：完整答案 DAG 共享判定；运行时四策略标签/权重；普通 KG 数据适配器；按索引取特征。
3. `ed0a756`：共享实验入口、配置和六卡任务清单。
4. `7b9c8f6`：只在完整答案 DAG 上统计路径数，省去与答案无关的节点遍历，统计定义不变。
5. `ab9c9f8`：有依赖的准备/训练任务调度；每个数据集准备好即开始训练。
6. `d444a9e`：跨数据集配对聚合、结构分桶、训练 PCN rank、按任务显存余量恢复失败运行。
7. 最终交付：补齐 MetaQA Reach@5 精确恢复和全最短路对照聚合；保存 110 次新增训练记录；导出逐种子 PCN 分布；Python 绘图与独立新稿、润色记录、文献核验和编译 PDF。全最短路后续清单拆为两个准备任务及十个可并发种子，避免原两条链的串行等待。

## 核心位置

`datasets/ordinary.py` 提供统一数据接口、MetaQA/KQAPro/PathQuestion 加载器、最短结构、实际采样和答案无关候选；超图适配沿用 `retriever_only/graph.py`，两者调用 `path_supervision/dag.py` 的同一判据。没有复制新的训练或损失实现。

`retriever_only/prepared.py::method_supervision` 返回标签副本和权重。PCN Mask 保持标签，Positive Relabel 只改 PCN，随机屏蔽逐题严格等量。`training.py` 使用原标签做共享候选级划分。

`scripts/run_experiment.py` 支持 encode、merge、prepare、train、evaluate、prevalence。原始结果存在时拒绝覆盖。每次训练保存参数、commit、命令、环境、起止时间、checkpoint 和逐题指标；数据版本及预处理边界在 prepared 元数据中。

`scripts/aggregate_cross_dataset_results.py` 统一旧逐题文件与新四策略结果。只有完整五种子才产生正式均值，不完整策略写 TODO；保存来源路径。`analyze_cross_dataset_mechanisms.py` 只在排名后用答案作结构分析。

## 具体正确性证据

新测试覆盖唯一最短路、另一条完整最短路、局部靠近答案但 prefix 绕远、屏蔽不改标签、重标记只改 PCN、随机数量、测试答案替换不影响候选/DDE/特征。按索引与物化特征的 CPU 训练在固定 batch 顺序下一 epoch checkpoint 完全相同。原训练/候选/评价相关测试已通过；新增聚合检查针对配对题覆盖和不完整种子，避免把部分结果当正式实验。

## 已遇到并处理的实际问题

KQA Pro 的 KB 字段为 `predicate`，实体输出终端为 `What`，按真实文件纠正适配。官方 test 缺标签，因此明确使用官方 val 子集。PathQuestion 发布文件缺可沿用的标准三分划分，按问题/topic 分组自定义划分并披露。

服务器访问 Hugging Face 下载和 GitHub fetch 出现超时；原始文件经本地下载后 SFTP 中转，阶段代码通过真实 Git bundle 同步，未改数据内容。三进程/卡使六卡持续接近满载，但大型物化张量导致若干 WikiTopics 任务 OOM；失败目录保留，独立 recovery 使用相同 FP32 特征的 indexed 读取，并等待显存余量。没有引入混合精度、扩大 batch 或缩减 epoch。
