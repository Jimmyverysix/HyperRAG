# WWW 修订记录

## 2026-09-20

- 从历史分支创建 `www-path-supervision-revision`，保留旧 audit、Gate C、加权确认、跨领域迁移与 proposal。
- 新增历史产物交叉校验器；原始 λ=0 Gate C 失败结论保持不变。
- 将完整主题—答案最短路条件和正式加权 BCE 拆分为独立模块。
- 正式损失统一为当前 batch 的 `sum(weight * BCE) / sum(weight)`。
- 冻结六点 λ 网格、问题级四分划分、单一 selection 指标和 test 隔离规则。
- 新增 330 任务 selection manifest、GPU 0--5 可恢复队列、冻结检查点 test 评估与两类诊断对照。
- 新增逐题 audit、dataset-specific λ、配对统计和相关分析的自动聚合链。
- 新增方法、实验协议、复现和语义人工审计文档。

## 2026-09-21

- 服务器通过 SSH 反向端口转发使用本机网络出口恢复 GitHub 同步；全程仍只经 GitHub 传递代码与纳入版本库的结果。
- 完成 330 个 selection、165 个冻结测试和 110 个 matched-random/all-positive 对照作业，失败数为 0。
- 完成 90,000 个问题的 11 领域路径冲突审计、逐题配对 bootstrap 和领域宏平均聚合。
- 路径一致方案相对基线的 answer reach@10 提高 6.89 个百分点，selected-path PR-AUC 下降 0.57 个百分点；相对匹配随机降权的覆盖提高 6.65 个百分点。
- 10/11 个领域选择 $\lambda=0$，tax 选择 $\lambda=0.1$；2/11 个领域的排序点估计下降超过 1 个百分点，因此结论写为“结构代理机制与宏平均折中得到验证”，而不是“逐领域无代价替换”。
- 正式结果及 provenance 由服务器提交 `f433d94`，本地重新生成带中文字体的论文图、自动表格和统一主文档。

仍未完成且不会写成已完成：official HyperRAG Retriever/QA、open-domain end-to-end 和真实双人语义标注；这些项目依赖尚未提供的官方资产或人工标注。
