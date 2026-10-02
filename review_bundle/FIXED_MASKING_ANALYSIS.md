# Fixed Masking 事后简化分析

## 分析身份

本分析是 **POST-HOC SIMPLIFICATION ANALYSIS**。项目在提出“能否用一个全局固定方法替代领域级 lambda 校准”之前已经访问过 test，因此本分析不能伪装成新的确认性协议。触发条件是在完全不读取 test 的 validation 领域等权分析中，`lambda=0.00` 为全局最优；该条件已经满足。

## 比较对象

1. Baseline：保留全部负监督；
2. Matched Random Masking：逐问题随机屏蔽与路径一致负例等量的已采样负例；
3. Fixed Path-Consistent Masking：全领域固定 `lambda=0.00`；
4. Tuned Strategy：各领域使用 validation 选择的 `lambda_D*`。

art、edu、health、infra、loc、people、sci、tax 的 `lambda_D*` 已是零，因此固定 masking 直接复用原正式 test report。award、org、sport 分别原选 `0.50`、`0.75`、`0.50`，为这三个领域另行运行 fixed masking 和 matched-random masking，共 `3 × 2 × 5 = 30` 个事后 report。

## 宏平均结果

| 方法 | APC-MRR（%） | Reach@10（%） | Reach@5（%） |
|---|---:|---:|---:|
| Baseline | 6.054 | 12.345 | 7.664 |
| Matched Random Masking | 6.077 | 12.397 | 7.707 |
| Fixed Path-Consistent Masking | 6.312 | 12.858 | 8.073 |
| Domain-specific Tuned Strategy | 6.335 | 12.961 | 8.137 |

## 配对差异

| 比较 | APC-MRR 差值（百分点） | 95% CI | Reach@10 差值（百分点） | 95% CI |
|---|---:|---:|---:|---:|
| Fixed − Baseline | +0.257 | [0.223, 0.292] | +0.514 | [0.443, 0.586] |
| Fixed − Matched Random | +0.235 | [0.201, 0.269] | +0.462 | [0.391, 0.533] |
| Fixed − Tuned | -0.023 | [-0.037, -0.010] | -0.102 | [-0.135, -0.070] |

统计单位为同一查询，先对五个种子等权平均，再在领域内重采样查询并对 11 个领域等权平均；共 10,000 次 paired bootstrap，seed 为 20260928。

## 领域异质性

固定 masking 相对 Baseline 的 APC-MRR 在 art、edu、health、infra、loc、sci 的 95% CI 高于零；award 的区间低于零；org、people、sport、tax 的区间跨零。相对 Matched Random 时，art、edu、health、infra、sci 的区间高于零，award 和 org 低于零，其余跨零。

sci 的收益最大，为相对 Baseline +1.108 个百分点；award 则为 -0.172 个百分点。这说明总体宏平均收益真实存在，但“每个领域都改善”不是数据支持的 claim。

## 简化判断

固定 masking 相对 tuned strategy 的劣势在统计上可分辨，但实质量级很小：APC-MRR 仅低 0.023 个百分点，Reach@10 仅低 0.102 个百分点；它分别保留 tuned strategy 相对 Baseline 的 91.65% 和 83.39% 收益。结合 validation 的全局最优为零，最终选择无领域超参数的固定 masking。

论文必须同时披露两点：固定 masking 显著优于 Baseline/Matched Random；它并非与 tuned strategy 严格等价，而是以很小的宏平均损失换取去除逐领域校准。

原始聚合结果见 `artifacts/final_revision/fixed_masking/fixed_masking_results.csv`、`fixed_masking.json` 和 `final_main_results.csv`。
