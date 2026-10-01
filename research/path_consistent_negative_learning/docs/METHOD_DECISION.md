# 最终方法决策

## 决策

最终正文采用 **Path-Consistent Negative Masking（路径一致负例屏蔽）**，即固定 `lambda=0.00`。它不增加模型参数、领域超参数或推理步骤。连续 lambda 形式只保留为连接 baseline 与 masking 的统一分析，完整网格移至附录。

这属于 Case A，但不是“数值严格相等”：固定 masking 相对领域级 tuned strategy 有很小、且因查询数较大而统计可分辨的下降。采用 masking 的理由是 validation 的全局最优、差异的实质量级，以及去掉领域校准后得到的可复现性和方法简洁性共同成立。

## 1. 固定 lambda 的 validation 最优值

领域等权 validation APC-MRR：

- `lambda=0.00`：6.1883%；
- `lambda=0.10`：6.0926%；
- `lambda=0.25`：6.0214%；
- `lambda=0.50`：5.9443%；
- `lambda=0.75`：5.9084%；
- `lambda=1.00`：5.8808%。

全局最优为 `lambda=0.00`，且 test 未参与该选择。

## 2. Fixed Masking 的 test 表现

下表均为 11 领域等权宏平均；差值使用同一查询配对、先跨五个种子平均、再进行 10,000 次 bootstrap。

| 比较 | APC-MRR 差值（百分点） | 95% CI | Reach@10 差值（百分点） | 95% CI |
|---|---:|---:|---:|---:|
| Fixed Masking − Baseline | +0.257 | [0.223, 0.292] | +0.514 | [0.443, 0.586] |
| Fixed Masking − Matched Random | +0.235 | [0.201, 0.269] | +0.462 | [0.391, 0.533] |
| Fixed Masking − Tuned Strategy | -0.023 | [-0.037, -0.010] | -0.102 | [-0.135, -0.070] |

固定 masking 对 Baseline 和 Matched Random 的两个指标区间都严格高于零。相对 tuned strategy 的区间严格低于零，所以不能声称统计等价；但最不利区间端点也只有 APC-MRR −0.037、Reach@10 −0.135 个百分点，远低于项目此前采用的 1 个百分点实质差异尺度。

固定 masking 保留了 tuned strategy 相对 Baseline 的 91.65% APC-MRR 收益和 83.39% Reach@10 收益，同时删除了领域级 lambda 选择。领域异质性必须保留：award 的固定 masking 明显弱于 tuned，org 也较弱；sport 的 APC-MRR 基本不变而 Reach@10 略低。其余 8 个领域本来就由 validation 选择 `lambda=0.00`。

## 3. 是否保留软加权

不作为最终主方法保留，原因是：

1. validation 的全局固定最优已经是 `0.00`；
2. 领域校准只在 3/11 个领域改变权重；
3. 它带来的宏平均增益只有 0.023 个 APC-MRR 百分点和 0.102 个 Reach@10 百分点；
4. 保留它需要额外的逐领域 validation 网格、选择文件和部署时领域身份。

软加权仍有分析价值：它表明 award、org 和 sport 存在领域异质性，并提供固定 masking 简化所牺牲性能的上界。论文不得删除这些数字，也不得把固定 masking 称为与 tuned strategy 严格等价。

## 4. 最终方法是否无额外超参数

是。最终方法只有一个由结构定义确定的二元操作：路径一致负例权重为零，其他样本权重为一。它不需要选择 lambda，不修改候选生成、模型、优化器或推理。

## 5. 论文中删除的旧 claim

- 删除 tie-breaking 作为独立研究故事、结果分支和贡献点；
- 删除“最优前进证书”“伪二元三元组”等不透明术语；
- 删除把早期结构代理结果与正式 Retriever-only 结果混写的表述；
- 删除任何暗示端到端生成质量、语义真值或第二数据集泛化已经得到验证的表述；
- 删除 fixed masking 与 tuned strategy 严格等价的暗示。

## 6. 第二数据集决策

本轮不扩张实验范围。现有结果足以支持 WikiTopics 11 个领域内的受控机制结论，但不能证明跨 benchmark 泛化。独立第二数据集是投稿前最有价值的外部效度增强项之一；只有在它提供兼容的主题实体、答案实体、训练图和可复算路径监督时才值得加入，否则会改变问题定义而不是复现实验。当前论文将这一点列为局限，不用不兼容数据集制造表面上的“第二数据集”。
