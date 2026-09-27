# 官方 HyperRetriever 服务器运行状态

更新时间：2026-09-27

## 1. 代码与历史隔离

- 历史工作树：`/root/hyperrag_pcneg/work/HyperRAG`，保留上一轮未提交图文件，不切换、不清理。
- 官方工作树：`/root/hyperrag_pcneg/work/HyperRAG_official`。
- 官方实验分支：`codex/www27-official-hyperretriever`。
- 当前研究提交：以 `git rev-parse HEAD` 为准；代码只经 GitHub 更新。

## 2. 数据与模型

- WikiTopics NLG：`/root/hyperrag_pcneg/data_sources/WikiTopicsQE_NLG`，121 个文件、11 个领域。
- 官方工作树数据入口：`dataset/WikiTopicsQE_NLG`，指向上述只读数据目录。
- GTE 标准缓存：`/root/hyperrag_pcneg/data_sources/hf_cache`。
- 已验证模型：`Alibaba-NLP/gte-large-en-v1.5` 及其 `Alibaba-NLP/new-impl` 代码。

正式命令固定：

```bash
export HF_HUB_CACHE=/root/hyperrag_pcneg/data_sources/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. Python/CUDA 环境

解释器：`/root/miniconda3/envs/hyperrag_official/bin/python`

- Python 3.11.16
- PyTorch 2.3.0+cu121
- Transformers 4.52.4
- NetworkX 3.5
- scikit-learn 1.7.0
- OpenAI Python 1.88.0
- CUDA 可用；机器可见 8 张 RTX 3090，研究只使用 0–5 号卡。

官方 `model.emb` 已在 `CUDA_VISIBLE_DEVICES=5` 下按官方模型名离线加载成功，输出 hidden size 1024、tokenizer vocab size 30522；进程已正常退出。

## 4. 当前唯一外部阻塞

没有发现 `OPENAI_API_KEY` 或 `config.json`。官方构图需要 `gpt-4o-mini` 和默认 `text-embedding-3-small`，训练数据准备再次需要 `gpt-4o-mini`，最终推理也需要 `gpt-4o-mini`。服务器不能直接访问 OpenAI 官方端点。

继续 baseline 前需要：

1. 一个有权调用 `gpt-4o-mini` 与 `text-embedding-3-small` 的 API key；
2. 若不是 OpenAI 官方端点，同时提供 OpenAI 兼容 base URL；
3. 该端点从服务器直接可达，或允许在正式运行期间通过本地代理转发。

凭据只放环境变量或被 Git 忽略的 `config.json`，绝不提交仓库。到位后先对单个请求做 chat 与 embedding smoke test，再只跑一个 WikiTopics 领域的完整 baseline 闭环。
