# 复现与服务器运行手册

在仓库根执行模块命令。服务器环境为 Python 3.11、PyTorch 2.3.0+cu121，模型目录 `/root/hyperrag_pcneg/data_sources/models/gte-large-en-v1.5`。本轮独立代码目录 `/root/hyperrag_pcneg/work/pcn_cross_dataset_20261003`，结果目录 `/root/hyperrag_pcneg/runs/cross_dataset_20261003`。不在文档、命令文件或代码中保存 SSH 密码。

## 数据与检查目的

服务器已保留本轮下载的原文件；严格复现优先复用它们。新环境可运行以下六线程下载，目标文件存在时跳过，不重新分发数据。PathQuestion 的单独数据许可未明确，KQAPro 数据 CC BY-SA 4.0；来源和使用边界见 `DATASET_SELECTION_ZH.md`。直接下载入口指向发布主分支，后续若上游变化，应与本轮保留原文件区分记录。

```bash
python - <<'PY'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve
root = Path('/root/hyperrag_pcneg/data_sources/pcn_cross_dataset_20261003')
sources = [(root/'pathquestion'/name,
            'https://raw.githubusercontent.com/zmtkeke/IRN/master/data/PQ/'+name)
           for name in ('2H-kb.txt', '3H-kb.txt', '2H.txt', '3H.txt')]
sources += [(root/'kqapro'/name,
             'https://huggingface.co/datasets/drt/kqa_pro/resolve/main/'+name)
            for name in ('kb.json', 'train.json', 'val.json', 'test.json')]
def download(item):
    path, url = item
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, path)
    return str(path)
with ThreadPoolExecutor(max_workers=6) as pool:
    print('\n'.join(pool.map(download, sources)))
PY
```

先运行 `inspect_dataset`，其目的是确认真实字段、topic/answer 接口、实际 PCN 和可用训练样本；失败时修正适配或停止该数据集，而不是填写估计数字。

```bash
python -m research.path_consistent_negative_learning.scripts.inspect_dataset \
  --dataset pathquestion --data-root /root/hyperrag_pcneg/data_sources/pcn_cross_dataset_20261003/pathquestion \
  --output /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion_inspection.json
```

将 dataset 改为 `kqapro`，data-root 改为对应目录可检查另一组。原始下载位置和适配定义见数据集选择文件；版本保留在 prepared 的 `dataset.json` 中。PathQuestion 四个文件来自作者 IRN 的 `data/PQ/`；KQAPro 四个 JSON 来自声明的镜像。服务器下载不可用时可从本地 SFTP 中转原文件，不改内容。

## 编码和准备

```bash
python -m research.path_consistent_negative_learning.scripts.build_cross_dataset_manifest \
  --phase encode-new --output /root/hyperrag_pcneg/runs/cross_dataset_20261003/manifests/encode-new.json
python -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
  --manifest /root/hyperrag_pcneg/runs/cross_dataset_20261003/manifests/encode-new.json \
  --gpus 0 2 3 4 5 6 --maximum-used-mib 12000
```

每组编码分为三个互不重叠的 shard，六卡并行。然后对每个 dataset 合并：

```bash
python -m research.path_consistent_negative_learning.scripts.run_experiment merge-embeddings \
  --inputs /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/encoding/0/embeddings.pt \
           /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/encoding/1/embeddings.pt \
           /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/encoding/2/embeddings.pt \
  --output /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/embeddings.pt
```

将路径中的 dataset 替换后合并 KQAPro。任务依赖保证 train/test 准备完成后才训练，不需要等另一组准备结束：

```bash
python -m research.path_consistent_negative_learning.scripts.build_cross_dataset_manifest \
  --phase main --output /root/hyperrag_pcneg/runs/cross_dataset_20261003/manifests/main.json
python -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
  --manifest /root/hyperrag_pcneg/runs/cross_dataset_20261003/manifests/main.json \
  --gpus 0 2 3 4 5 6 --maximum-used-mib 12000 --workers-per-gpu 3
```

清单含六项准备、40 次新基准四策略训练和 60 次既有基准正例重标记。旧三策略不重跑；主清单中 WikiTopics 正例重标记现在用 indexed 存储以支持同卡并发。物理卡数量最多六张，卡号不限于 0–5；不要占用未授权或其他任务的卡。已完成 expected files 的任务跳过。

## 单次训练和可选构造

```bash
python -m research.path_consistent_negative_learning.scripts.run_experiment train \
  --prepared /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/prepared/train_seed_42.pt \
  --embeddings /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/embeddings.pt \
  --evaluation /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/prepared/test.pt \
  --output-dir /root/hyperrag_pcneg/runs/cross_dataset_20261003/pathquestion/pcn_mask/seed_42 \
  --strategy pcn_mask --seed 42 --batch-size 512 --feature-storage indexed
```

策略可取 `baseline`、`random_mask`、`pcn_mask`、`positive_relabel`；共享同种子 prepared 文件，不为每个策略重采样。已有 checkpoint 或 metrics 时拒绝覆盖，另选明确的新输出目录。

```bash
python -m research.path_consistent_negative_learning.scripts.build_cross_dataset_manifest \
  --phase all-shortest --output /root/hyperrag_pcneg/runs/cross_dataset_20261003/manifests/all-shortest.json
python -m research.path_consistent_negative_learning.scripts.run_retriever_queue \
  --manifest /root/hyperrag_pcneg/runs/cross_dataset_20261003/manifests/all-shortest.json \
  --gpus 0 2 3 4 5 6 --maximum-used-mib 24000 --workers-per-gpu 3
```

该清单含两项独立 prepare `--all-shortest` 及十项有依赖的单种子训练；一组准备好后其五种子即可并行，不串行占一张卡。evaluation 池复用四策略同一文件。配置中训练方法显示 baseline 权重、`source_protocol=all_shortest_positive_union_new_negative_pool`，输出策略目录才是 all_shortest；它不被误认为主表 Original。已完成这一对照的旧清单是两条 prepare+五种子链，其真实日志保留；新清单只改未来调度，不重跑或改写现有结果。

## OOM 恢复及日志

本轮加速曾在大型物化张量上 OOM。原失败目录保留，成功恢复位于 `recovery_indexed/`；它们只改变相同特征的内存读取方式。恢复清单每任务 `maximum_used_mib=18000`，命令启动前等待余量；六卡启动筛选可用 24000，以免把尚在工作的卡永久排除。该余量控制针对本轮真实 OOM，不改变实验参数。

查看 `command_0.stdout.log` 的 epoch 和 loss、`command_0.stderr.log` 的具体错误，决定继续、恢复或修复。队列 `.complete` 表示该清单成功；中途停止主调度器后，原子进程可能继续完成，所以全轮状态应以每次训练的 metrics/config/checkpoint 及最终聚合为准。不要杀掉无关进程。

## 汇总、分析与论文

```bash
python -m research.path_consistent_negative_learning.scripts.aggregate_cross_dataset_results \
  --output-dir /root/hyperrag_pcneg/runs/cross_dataset_20261003/aggregates
python -m research.path_consistent_negative_learning.scripts.analyze_cross_dataset_mechanisms \
  --dataset pathquestion --output /root/hyperrag_pcneg/runs/cross_dataset_20261003/aggregates/pathquestion_mechanisms.json
```

第二条改为 kqapro 可分析另一组，使用 CPU 对真实训练 PCN 评分，不增加第七张 GPU。聚合只有五种子齐全才报策略均值；配对题集、Oracle 和候选数量不一致时停止，不生成混合结果。MetaQA 的 Reach@5 从逐题 RR≥1/5 恢复，并与旧每种子 JSON 核对。

本地同步小型 JSON/CSV 后运行以下命令，生成分布、图和新 `.tex`，不覆盖旧稿。单文件源内嵌 Python 导出的 PGF 图；Codex 内置编译器当前因系统标准目录不可用而失败，本轮由本机既有 XeLaTeX 导出 PDF。润色前版本及具体修改记录独立保存。

```bash
python -m research.path_consistent_negative_learning.scripts.summarize_cross_dataset_prevalence
python -m research.path_consistent_negative_learning.scripts.build_restructured_paper
xelatex -interaction=nonstopmode -halt-on-error -output-directory=research/path_consistent_negative_learning/paper/restructured_zh/build paper_restructured_zh.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=research/path_consistent_negative_learning/paper/restructured_zh/build paper_restructured_zh.tex
```

首次编译前创建上述 build 目录。本地 `.tex` 单独编译不依赖作图工具；重建图需要 matplotlib、NumPy、既有 XeLaTeX 和 TeX Gyre Heros 字体。完整表、绘图源数据和统计分母以汇总文件为准。
