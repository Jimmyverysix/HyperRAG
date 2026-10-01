# Lambda 简化分析

## 结论

只使用 validation、按 11 个领域等权宏平均时，固定 `lambda=0.00` 的 APC-MRR 最高。因此，验证证据支持把连续软抑制简化为固定路径一致负例屏蔽；这一判断没有读取 test。

## 固定 lambda 的 validation 结果

| lambda | 领域等权 APC-MRR（%） | 相对 `lambda=0.00`（百分点） |
|---:|---:|---:|
| 0.00 | 6.1883 | 0.0000 |
| 0.10 | 6.0926 | -0.0957 |
| 0.25 | 6.0214 | -0.1669 |
| 0.50 | 5.9443 | -0.2440 |
| 0.75 | 5.9084 | -0.2799 |
| 1.00 | 5.8808 | -0.3075 |

从 `0.00` 到 `1.00` 的宏平均整体下降。最接近的备选 `0.10` 仍低 0.0957 个百分点，因此全局结论不是由数值并列规则决定。

## 领域级选择与宏平均选择的区别

原确认性协议按领域在 validation 上选择 `lambda_D*`。11 个领域中：

- 8 个选择 `0.00`：art、edu、health、infra、loc、people、sci、tax；
- award 和 sport 选择 `0.50`；
- org 选择 `0.75`。

领域级校准回答“每个领域的最佳网格点是什么”，领域等权分析回答“若只允许一个全局方法，哪个固定权重最好”。二者没有数据冲突；后者选择 `0.00`，为无领域超参数的简化提供 validation 依据。

## 数据隔离

- 输入只来自 330 个 validation report，即 `11 领域 × 6 个 lambda × 5 个种子`。
- 每个领域先对五个种子求算术均值，再对领域等权平均。
- `selection/lambdas.json` 明确记录 `test_metrics_accessed=false`。
- test 只在该判断冻结后用于标记为事后的固定 masking 简化分析。

原始聚合结果见 `artifacts/final_revision/lambda_validation/lambda_validation.csv` 和同目录 JSON。
