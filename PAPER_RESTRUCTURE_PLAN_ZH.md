# 新中文稿的论证与重构计划

使用 nature-writing 完成证据驱动初稿，再用 nature-polishing 做独立修订。轴为 research / 全文 / 中文 / generic；用户要求中文优先于语言片段中的默认英译。主读者为 IR、结构化检索和 KGQA 研究者。

**一句话论证：** 在单路径构造的结构化检索弱监督中，实际负采样会纳入其他完整最短答案路径的转移；精确识别并撤销这部分负监督，在四组明确界定的数据上改善 APC-MRR，但结构一致性不等同于语义正例，屏蔽与重标记无统一优劣，重建全最短路监督在两个新基准上更好。

## 标题候选

| 中文 | 英文 |
|---|---|
| 路径一致负例：单路径结构化检索中的监督冲突与修正 | Path-Consistent Negatives: Supervision Conflicts and Correction in Single-Path Structured Retrieval |
| 当答案路径进入负采样：结构化检索的监督一致性 | When Answer Paths Enter Negative Sampling: Supervision Consistency in Structured Retrieval |
| 未被选择的路径不等于负例：路径弱监督的结构冲突 | An Unselected Path Is Not a Negative: Structural Conflicts in Path-Based Weak Supervision |
| 从路径选择到监督撤销：识别结构化检索中的冲突负例 | From Path Selection to Supervision Abstention: Identifying Conflicting Negatives in Structured Retrieval |
| 多条答案路径，一条正监督：结构化检索中的负例冲突 | Multiple Answer Paths, One Positive Trace: Conflicting Negatives in Structured Retrieval |
| 完整答案路径上的负标签：结构证据及其监督边界 | Negative Labels on Complete Answer Paths: Structural Evidence and Its Supervisory Limits |

选第一题：突出问题、定义与修正，不承诺屏蔽击败所有替代方案。

## 章节工作与证据映射

1. 引言：结构检索的监督需求 → 选择之外即负例的隐含假设 → 两条完整最短路反例 → 四组真实 prevalence → 最小撤销 → 有边界的跨数据证据。贡献按问题、形式化、最小修正、实证组织。
2. 单路径中的结构冲突：图/超图、每个答案的一条所选路径、实际负采样集合；区分潜在替代路径与实际 PCN。
3. PCN 形式化：完整距离等式、答案最短路 DAG 等价实现；拒绝局部距离缩短判据。给出图结构证据与语义正例证据的界限。
4. 最小一致性修正：共享加权 BCE、四种 y/w、随机数量控制、重标记；训练修正与推理分开。
5. 实验设计：六个 RQ、数据来源与投影、共享控制、指标分母、配对统计。
6. 现象：跨数据 prevalence 与逐题分布，区分五种子样本总数、任一种子受影响题数。
7. 主结果：四策略绝对分数、三项差值 CI、可达条件结果；对重标记优势和未区分结果如实描述。
8. 机制：路径数、跳数、关系多样性、实际训练 topic PCN 密度、candidate pool、领域和训练 PCN rank；描述关联，不写因果证明。
9. 相关工作：弱监督路径选择、难负例与伪负例、鲁棒标签学习和监督弃权、KG/超图检索；指出本文检验对象是实际采样交集及完整路径条件，而非新增搜索器。
10. 讨论与结论：结构证据可否定负标签推导，但不能赋予完整语义解释；最短路、受限子集、候选上限、候选级早停、有限模型和五种子条件 CI 的边界。

写作顺序为结果 → 引言/结论 → 方法/讨论 → 标题/摘要。新建全文，不 patch 旧稿。旧结果仅提供可核对数字；旧版本中的废弃历史说法不进入新稿。

## 术语与符号台账

| 统一写法 | 定义/限制 |
|---|---|
| 路径一致负例（Path-Consistent Negative, PCN） | 实际采样负例 ∩ 完整最短答案路径转移并集 |
| 结构监督冲突 | 负标签仅源自未被选择，却具有同一结构标准下的答案路径身份；不是语义真假定理 |
| PCN Mask / PCN 屏蔽 | 标签 y=0 不变，w=0 |
| Positive Relabel / 正例重标记 | 仅实际 PCN 改 y=1，w=1 |
| All-shortest / 全最短路监督 | 在构造阶段改变正例并重新采样，不能等同于重标记 |
| APC-MRR | 排序前缀形成完整路径时的倒数排名均值 |
| Candidate Oracle | 固定候选池中至少有一条完整答案路径 |
| D_q | 实际 PCN 集；不用于推理 |
| S_q / N_q / U_q | 所选正转移 / 实际负转移 / 全部最短答案路径转移并集 |

## 主张—证据—边界

“PCN 非单一超图特例”：四组训练真实采样计数支持，不能外推全部 KGQA。“屏蔽改善排序”：五种子 APC-MRR 与三对照 CI 支持程度分别报告，Reach 指标可以不显著。“屏蔽优于重标记”：不能先验成立，KQAPro 均值提供相反方向证据。“全最短路是否更好”：两个新基准的完成结果更高，须在摘要、引言和结论中约束最小修正的价值，不能只列可选表。“覆盖与排序不同”：候选固定、Oracle 与条件指标支持，不能用重排恢复缺失路径。“机制”：分桶和 rank 支持描述，不能排除更多训练正则化解释。
