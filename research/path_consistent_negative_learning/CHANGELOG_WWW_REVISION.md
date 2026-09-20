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

未完成项不会在本记录中写成已完成：official HyperRAG Retriever/QA、open-domain end-to-end、真实双人语义标注和本轮新 GPU 结果仍依赖外部资产或服务器 GitHub 网络恢复。
