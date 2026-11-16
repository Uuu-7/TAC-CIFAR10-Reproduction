# TAC 论文复现：CIFAR-10 无训练版本

复现 [Li 等，*Image Clustering with External Guidance*，ICML 2024](https://proceedings.mlr.press/v235/li24ar.html) 的 CIFAR-10 **TAC (no train)** 实验。论文还包含双分支蒸馏训练的完整 TAC；本仓库当前结果不代表该完整模型。

## 已复现结果

论文表 2 的列顺序为 NMI / ACC / ARI。下表统一按 ACC / NMI / ARI 显示，单位为百分比。复现值为 5 个 K-means 种子的均值；完整逐次结果见 [`runs.csv`](reproduction/results/runs.csv)，配置与环境见 [`summary.json`](reproduction/results/summary.json)。

| 方法 | ACC | NMI | ARI |
|---|---:|---:|---:|
| CLIP K-means，论文 | 74.20 | 70.30 | 61.60 |
| CLIP K-means，复现 | 75.15 | 71.30 | 62.57 |
| TAC no train，论文 | 90.10 | 80.80 | 79.80 |
| TAC no train，复现 | 89.18 | 79.67 | 78.05 |

复现使用 OpenAI CLIP ViT-B/32、CIFAR-10 的 50,000 张训练图像构建文本空间，并在 10,000 张测试图像上评估。167 个语义中心，每中心至多 5 个 WordNet 名词，检索温度 0.005。名词提示词采用论文描述的 `a photo of a {}.`。聚类采用 scikit-learn CPU 欧氏 K-means 近似作者代码的 FAISS GPU 球面 K-means，因此数字不应被视为精确复现。该近似是目前主要的协议差异。

![论文与复现对比](reproduction/results/paper_vs_reproduction.png)

## 运行

在 `reproduction/` 目录下：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python reproduce_cifar10.py --data-dir data --output-dir results --prompt-set paper-single --noun-source full-wordnet
python plot_paper_vs_reproduction.py
```

运行前将作者项目的 `WordNetNouns.csv` 放入 `reproduction/data/`。CIFAR-10 和 CLIP 权重在首次运行时下载，文件与中间特征保存在 `data/`，并被 Git 忽略。CPU 上首次运行可能耗时较长；中断后可复用已完成的嵌入缓存。更改 WordNet 文件、CLIP 权重或图像预处理时，应清空相关缓存后重跑。

`--prompt-set authors-seven` 使用作者公开代码中的七个提示词。此选项会重新计算文本嵌入与名词筛选，不会混用单提示词的筛选缓存。`--noun-source paper-top30` 仅用于快速流程检查，不等同于完整 WordNet 实验。

## 改进实验：文本权重

论文把单位化图像和检索文本向量等权拼接。`fusion_ablation.py` 在缓存的测试嵌入上扫描文本权重 `0.5, 0.75, 1.0, 1.25, 1.5`，其中 `1.0` 对应当前基线。运行：

```powershell
python fusion_ablation.py
```

输出 `results/fusion_ablation.csv`。这是**探索性消融**：测试标签只用于计算指标，不能根据这些测试结果挑选权重后声称泛化提升。若某个权重较好，下一步应在独立数据集或预先固定的验证协议中确认。

| 文本权重 | ACC | NMI | ARI |
|---:|---:|---:|---:|
| 0.75 | 89.29 | 80.10 | 78.17 |
| 1.00（论文等权基线） | 89.18 | 79.67 | 78.05 |

在这份 CIFAR-10 测试集上的探索结果中，0.75 的三项均值分别高出约 0.11、0.43、0.12 个百分点；这是待独立验证的观察结果。

## 代码来源与局限

复现参照了[作者公开实现](https://github.com/XLearning-SCU/2024-ICML-TAC)；作者原始模块只保留在本地，未上传到本仓库。这里发布的 `reproduce_cifar10.py`、`plot_paper_vs_reproduction.py` 和 `fusion_ablation.py` 是独立脚本。作者原始训练脚本依赖 CUDA/FAISS GPU，本仓库未验证完整训练版 TAC，也未发布数据、模型权重或论文 PDF。
