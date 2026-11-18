# TAC Reproduction: Training-Free CIFAR-10

This repository reproduces the CIFAR-10 **TAC (no train)** experiment from [Li et al., *Image Clustering with External Guidance*, ICML 2024](https://proceedings.mlr.press/v235/li24aa.html). It also includes scripts for stability analysis, prompt and component ablations, transfer experiments, and full TAC distillation. The long transfer and training runs still need to be executed; see [Additional experiments](EXPERIMENTS.md) for exact PowerShell commands and current results.

## Reproduction results

Table 2 of the paper lists NMI / ACC / ARI. The table below uses ACC / NMI / ARI throughout. Scores are percentages; reproduced scores are means over five K-means seeds. See [`runs.csv`](reproduction/results/runs.csv) for individual runs and [`summary.json`](reproduction/results/summary.json) for the configuration and environment.

| Method | ACC | NMI | ARI |
|---|---:|---:|---:|
| CLIP K-means, paper | 74.20 | 70.30 | 61.60 |
| CLIP K-means, reproduced | 75.15 | 71.30 | 62.57 |
| TAC (no train), paper | 90.10 | 80.80 | 79.80 |
| TAC (no train), reproduced | 89.18 | 79.67 | 78.05 |

The reproduction uses OpenAI CLIP ViT-B/32 and 50,000 CIFAR-10 training images to construct the text space. It evaluates on 10,000 test images. It uses 167 semantic centers, up to five WordNet nouns per center, a retrieval temperature of 0.005, and the paper's `a photo of a {}.` prompt. Clustering uses scikit-learn CPU Euclidean K-means as an approximation to the authors' FAISS GPU spherical K-means. This is the main protocol difference, so these scores should not be considered an exact reproduction.

![Paper results compared with the reproduction](reproduction/results/paper_vs_reproduction.png)

## Run the experiment

From the `reproduction/` directory, on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python reproduce_cifar10.py --data-dir data --output-dir results --prompt-set paper-single --noun-source full-wordnet
python plot_paper_vs_reproduction.py
```

Before running, place `WordNetNouns.csv` from the [authors' repository](https://github.com/XLearning-SCU/2024-ICML-TAC) in `reproduction/data/`. CIFAR-10 and the CLIP weights are downloaded on the first run. Downloads and intermediate embeddings remain under `data/` and are ignored by Git. The first CPU run can take a long time; completed embedding caches are reused after an interruption. If you change the WordNet file, CLIP weights, or image preprocessing, clear the affected caches before rerunning.

`--prompt-set authors-seven` uses the seven prompts in the authors' released code. It recomputes text embeddings and noun selection without reusing the single-prompt selection cache. `--noun-source paper-top30` is a quick pipeline check using the paper's published nouns, not the full WordNet experiment.

### FAISS spherical K-means and seven-prompt run

Install the optional FAISS dependency with `pip install -r requirements-faiss.txt`. Then run the following from `reproduction/`:

```powershell
python reproduce_cifar10.py --data-dir data --output-dir results/faiss_single --prompt-set paper-single --noun-source full-wordnet --kmeans-backend faiss-cpu --threads 8
python reproduce_cifar10.py --data-dir data --output-dir results/faiss_seven --prompt-set authors-seven --noun-source full-wordnet --kmeans-backend faiss-cpu --trim-text-padding --threads 8
```

The seven-prompt run encodes the full WordNet vocabulary and can take hours on a CPU. It writes a partial array and progress file after every text batch. Repeating the same command resumes the current prompt. `--trim-text-padding` skips positions after CLIP's end-of-text token; because the attention is causal, those positions cannot affect the returned embedding. A 256-noun check showed a maximum absolute difference of `2.4e-6` from the standard encoder. The optimized run uses separate caches from the standard encoder.

Both FAISS CPU runs are complete. The full WordNet noun selection yielded 776 nouns with one prompt and 749 with seven prompts. The table reports image-plus-text mean ± sample standard deviation over seeds 0–4, with 20 FAISS restarts per seed:

| Prompt set | ACC | NMI | ARI |
|---|---:|---:|---:|
| Single prompt, [`runs`](reproduction/results/faiss_single/runs.csv) / [`summary`](reproduction/results/faiss_single/summary.json) | 84.16 ± 6.79 | 78.35 ± 2.03 | 74.28 ± 4.87 |
| Seven prompts, [`runs`](reproduction/results/faiss_seven/runs.csv) / [`summary`](reproduction/results/faiss_seven/summary.json) | 83.67 ± 6.42 | 78.86 ± 2.10 | 74.52 ± 4.99 |

The seven-prompt ensemble slightly raised mean NMI and ARI but did not raise mean ACC. The large seed-to-seed variation prevents a reliable improvement claim. These are FAISS CPU results; the authors used FAISS GPU. Both runs use the same clustering backend and seeds, and each prompt set selects its own nouns from the full WordNet vocabulary. A [20-seed stability diagnostic](EXPERIMENTS.md#1-faiss-stability-and-sampling-diagnostic) shows that FAISS's local default sample cap is a material source of variation.

## Exploratory improvement: text fusion weight

The paper concatenates unit-normalized image and retrieved-text features with equal weights. `fusion_ablation.py` evaluates text weights `0.5, 0.75, 1.0, 1.25, 1.5` on cached test embeddings; `1.0` is the equal-weight baseline. Run:

```powershell
python fusion_ablation.py
```

Results are saved to `results/fusion_ablation.csv`. This is an **exploratory ablation**. Test labels are used only to calculate metrics; selecting a weight based on this test set does not establish a generalization gain. A promising weight should be confirmed on an independent dataset or with a validation protocol fixed in advance.

| Text weight | ACC | NMI | ARI |
|---:|---:|---:|---:|
| 0.75 | 89.29 | 80.10 | 78.17 |
| 1.00 (equal-weight baseline) | 89.18 | 79.67 | 78.05 |

On this CIFAR-10 test set, weight 0.75 increased the three means by about 0.11, 0.43, and 0.12 percentage points, respectively. This observation still needs independent validation.

## Code provenance and limitations

The reproduction follows the [authors' public implementation](https://github.com/XLearning-SCU/2024-ICML-TAC). Local copies of the authors' original modules were not uploaded to this repository. The published scripts are standalone. The original training code requires CUDA and FAISS GPU; this repository's training adaptation uses FAISS CPU neighbors and supports PyTorch CPU. Its full trained TAC result remains unverified until the long run completes. Datasets, model weights, and paper PDFs are not distributed here.
