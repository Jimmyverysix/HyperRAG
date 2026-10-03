# 新稿文献核验

按 nature-academic-search 的多来源检索和引用核验流程执行。当前没有可调用的文献数据库 MCP，使用原始出版页、ACL Anthology、NeurIPS Proceedings、作者论文的 arXiv 页面和 PVLDB 原文核对。核验对象为新稿实际引用的十篇文献；没有把旧稿参考文献全量搬入。

| 键 | 原始来源与已核对信息 | 新稿用途 |
|---|---|---|
| hyperrag | [arXiv:2602.14470](https://arxiv.org/abs/2602.14470)，Wen-Sheng Lien 等，2026，*HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation* | 继承的超图检索框架；不混同其他同名 Hyper-RAG 项目 |
| metaqa | [AAAI 原始页面](https://ojs.aaai.org/index.php/AAAI/article/view/12057)，Yuyu Zhang、Hanjun Dai、Zornitsa Kozareva、Alexander Smola、Le Song，2018，DOI 10.1609/aaai.v32i1.12057 | MetaQA 与路径推理任务来源 |
| pq | [ACL Anthology C18-1171](https://aclanthology.org/C18-1171/)，Mantong Zhou、Minlie Huang、Xiaoyan Zhu，COLING 2018，2010–2022 | IRN 与 PathQuestion 来源 |
| kqa | [ACL Anthology 2022.acl-long.422](https://aclanthology.org/2022.acl-long.422/)，Shulin Cao、Jiaxin Shi、Liangming Pan 等，ACL 2022，6101–6119 | KQA Pro 数据与程序标注；不把本文子集当完整任务 |
| dpr | [ACL Anthology 2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/)，Vladimir Karpukhin 等，EMNLP 2020，6769–6781 | 密集检索的负例训练背景 |
| ance | [arXiv:2007.00808](https://arxiv.org/abs/2007.00808)，Lee Xiong、Chenyan Xiong、Ye Li 等，2020 | 近邻难负例；核对并纠正首作者为 Lee Xiong，引用此预印本年份 |
| debiased | [NeurIPS 原始页面](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)，Ching-Yao Chuang 等，NeurIPS 33，2020 | 负采样可能含同类样本的目标修正；不声称使用其算法 |
| false_negative | [AAAI DOI](https://doi.org/10.1609/aaai.v38i17.29885)，Shiqi Wang、Yeqin Zhang、Cam-Tu Nguyen，2024，38(17):19171–19179 | 密集检索的伪负例与置信度正则化 |
| gce | [NeurIPS 原始页面](https://proceedings.neurips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)，Zhilu Zhang、Mert R. Sabuncu，NeurIPS 31，2018 | 噪声标签鲁棒损失背景 |
| snorkel | [PVLDB 原始 PDF](https://www.vldb.org/pvldb/vol11/p269-ratner.pdf)，Alexander Ratner 等，2017，11(3):269–282，DOI 10.14778/3157794.3157797 | 弱监督来源与冲突；不声称 PCN 是 Snorkel 标签模型 |

新稿以普通摘要性陈述引用，没有长段原文复制。文献支持相关工作定位，本文的 PCN 发生比例、排序改善、训练规模和机制分析全部来自本项目结果，不由这些引用背书。未找到覆盖相同研究对象的文献不等于证明“首次”；新稿不作该声明。

独立 BibTeX 保存为 `references_restructured.bib`。为适配 Codex 单文件 LaTeX 编辑器，根 `.tex` 内保留相同十项 `thebibliography`，编译不依赖外部 BibTeX 文件。旧稿的 BibTeX 与用户已有改动未被覆盖。
