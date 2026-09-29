# Retriever-only 无损加速记录

## 问题定位

路径择一敏感性原先有 11 个领域、3 个路径变体和 5 个训练种子，共 165 次候选预处理。对同一“领域 × 路径变体”，最短路径选择、路径引导子图和最短路径 DAG 与训练种子无关，却被重复计算五次。预处理主要受 Python/NetworkX 单核图遍历限制；原队列仍按“一卡一进程”分配，导致进程尚未进入轻量 DDE 计算时就长期占有一个 GPU 调度名额。

## 实现

- `retriever_only/candidates.py` 将单题准备拆成种子无关的结构构建和种子相关的负例抽样。
- `retriever_only/preparation.py` 为五个种子保留各自独立的 `random.Random(seed)` 流，共享图结构计算，并缓存图的有序邻接表。
- `retriever_only/graph.py` 在目标第一次被 BFS 发现、最短路径父节点已经固定时立即返回，不再等待目标从队列弹出；返回路径与原实现逐条一致。
- `scripts/prepare_retriever_training_batch.py` 一次产生同一领域、同一路径变体的五份 `training.pt` 和独立 report。
- `scripts/build_retriever_manifest.py` 提供 `path-sensitivity-prepare`（33 个作业）和 `path-sensitivity-train`（330 个训练/评估作业），让准备与训练分别按空闲 GPU 动态调度。
- `scripts/run_retriever_queue.py` 支持 `--workers-per-gpu`。预处理在每张空闲卡上运行六个独立槽，以并行利用 CPU；训练按领域峰值显存分级调度。
- `retriever_only/training.py` 直接生成与冻结 DataLoader 完全相同的批索引，省去完整排列转 Python 列表、逐样本 `TensorDataset` 访问和逐批 collate；batch size、随机数消耗、批顺序及参数更新顺序均不变。
- 训练和验证的逐批 loss/权重标量先按原顺序写入 GPU 缓冲区，每个 epoch 结束时一次性传回 CPU，再按原来的 Python `float` 顺序累加。这样消除了每批多次 CPU--GPU 强制同步，同时保持早停数值和最佳 checkpoint 判定不变。

没有修改训练 batch size、网络、损失、lambda、种子、候选定义、测试集合或聚合口径。

## 等价性与速度

本地测试逐字段比较批量与原逐种子路径；服务器又在真实 `art` 数据上比较种子 42 和 43。候选、索引、标签、路径一致掩码和查询边界完全相同。DDE 继续按原来的逐题顺序计算；不同 CUDA 执行上下文之间观测到的最大差异为 `5.960464477539063e-08`，即一个 `float32` ULP，属于归约舍入噪声。

服务器以 `art` 前 20 个有效训练问题、五个种子进行端到端准备基准：原路径 85.314 秒，批量路径 15.000 秒，加速 5.688 倍。这个基准主要衡量被消除的重复图计算，不用于推断模型指标。

在同一服务器负载下，对 `art` 前 20 个有效问题单独剖析结构构建：等待目标出队的旧 BFS 为 44.860 秒，目标首次发现即返回的新 BFS 为 19.299 秒，加速 2.325 倍。测试以旧实现作为判定器，对三张随机图、三个正式路径变体种子及所有源--目标组合逐条比较，所选路径完全一致。

以 67.7 万个训练索引、batch size 32 模拟 `award` 的一个 epoch，旧 DataLoader 索引迭代为 3.983 秒，直接批索引为 0.397 秒，索引层加速 10.025 倍。多 epoch 测试逐批比较索引完全一致；完整小模型训练的每轮 loss 历史和 checkpoint 参数张量也逐位一致。该优化只消除 Python 数据管线开销，不改变矩阵运算。

## 运行边界

服务器有 64 个逻辑核和 251 GiB 内存。单个预处理进程实测约占 0.8 GiB 内存、一个 CPU 核，DDE 只构造小型传播张量，因此在三张空闲 3090 上使用 `6 × 3 = 18` 个预处理槽仍有充足 CPU、内存和显存余量。训练特征矩阵按候选数线性占用显存：`award` 实测单作业约 13.8 GiB，`health` 估算约 11 GiB，二者保持每卡一个作业；`art/edu/infra/loc/org/people/sci/sport/tax` 单作业约 4--7 GiB，每张 24 GiB 3090 运行三个独立作业。重、轻领域分别生成 manifest，前者 `--workers-per-gpu 1`，后者 `--workers-per-gpu 3`。

分级调度没有合并作业，也没有改变 batch size、随机数流或浮点运算顺序。队列仍逐作业检查四个预期产物并跳过完整结果；最终聚合前用原始 330 作业清单核对全集。2026-09-29 的正式敏感性恢复运行中，`art` 的 30 个作业已在切换前全部完成，因此轻领域恢复清单只含其余八个领域；该次运行只使用 GPU 1--3，避免干扰 GPU 0/4/5 上已有任务。中断切换前未完成的三个日志目录保存在运行根目录的 `archives/resource_schedule_switch_*` 下。

只调度启动时显存占用不超过阈值的 GPU，不终止其他任务。完整输出会自动跳过；不完整的历史输出不会被静默覆盖。每个槽只运行 manifest 中一个具有独立输出目录的作业，正式结果仍由聚合器从独立训练和评估 report 读取，科研 provenance 保持不变。
