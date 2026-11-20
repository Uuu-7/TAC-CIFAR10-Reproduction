# TAC Reproduction: CIFAR-10

This repository reproduces CIFAR-10 **TAC (no train)** and full TAC distillation from [Li et al., *Image Clustering with External Guidance*, ICML 2024](https://proceedings.mlr.press/v235/li24aa.html). Completed experiments also include stability analysis, prompt and component ablations, and CIFAR-20/STL-10 transfer. See [Additional experiments](EXPERIMENTS.md) for protocols, commands, and results.

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

## Additional experiments: current status

The five follow-up experiments use the same cached CIFAR-10 CLIP features and WordNet embeddings where applicable. **Completed** means numerical results are committed; **ready to run** means the implementation is committed but the long job has not been executed. Detailed protocols, commands, and resume instructions are in [EXPERIMENTS.md](EXPERIMENTS.md).

| Experiment | Status | Current result or output |
|---|---|---|
| 1. FAISS seed and sampling stability | Completed | [20-seed default-cap runs](reproduction/results/stability_default.csv): single-prompt ACC 85.00 ± 5.59; seven-prompt ACC 85.98 ± 5.78. [Five-seed full-data diagnostic](reproduction/results/stability_full_data.csv): 89.11 ± 0.01 and 88.84 ± 3.04, respectively. |
| 2. Separate noun selection from prompt encoding | Completed for FAISS CPU | [Four-cell experiment](reproduction/results/factorial_faiss.csv): single-prompt noun selection with seven-prompt retrieval encoding gives ACC 87.88 ± 5.26 over five seeds. The scikit-learn selection extension is available but has not been run. |
| 3. Component and hyperparameter ablations | Completed: 12 configurations plus image-only control | [Summary](reproduction/results/ablation_grid/summary.csv): five seeds per configuration. Gamma 20 gives ACC 90.21; baseline 88.84. [Image-only](reproduction/results/ablation_image_only.csv) gives ACC 75.13 ± 0.03 with matching FAISS evaluation settings. Full-WordNet retrieval remains unrun. |
| 4. CIFAR-20 and STL-10 transfer | Completed | Text weight 1.0 → 0.75 changes mean ACC from 54.39 → 55.04 on [CIFAR-20](reproduction/results/cifar20_transfer/summary.json), and 97.43 → 98.00 on [STL-10](reproduction/results/stl10_transfer/summary.json). CIFAR-20 NMI decreases. |
| 5. Full TAC distillation | Completed: 20 seeds, 20 epochs each | [20-seed summary](reproduction/results/tac_training_20seeds/summary.json): ACC 90.81 ± 2.76, NMI 82.61 ± 1.49, ARI 81.84 ± 2.96. Seeds 1 and 10 have ACC 83.28 and 82.23; all seeds are retained. The original [five-seed result](reproduction/results/tac_training/summary.json) is preserved. |

The fixed gamma=20 follow-up also completed on both transfer datasets. Compared within each text weight, gamma=20 increases all three mean metrics relative to gamma=5. These are five-seed observations on two datasets, not a claim of universal or statistically significant improvement:

| Dataset | Text weight | ACC, gamma 5 → 20 | NMI, gamma 5 → 20 | ARI, gamma 5 → 20 |
|---|---:|---:|---:|---:|
| [CIFAR-20](reproduction/results/cifar20_transfer_gamma20/summary.json) | 1.0 | 54.39 → 56.68 | 59.58 → 60.40 | 40.74 → 42.74 |
| CIFAR-20 | 0.75 | 55.04 → 56.63 | 59.07 → 59.18 | 40.74 → 42.23 |
| [STL-10](reproduction/results/stl10_transfer_gamma20/summary.json) | 1.0 | 97.43 → 98.28 | 94.12 → 95.65 | 94.41 → 96.23 |
| STL-10 | 0.75 | 98.00 → 98.28 | 95.08 → 95.73 | 95.63 → 96.24 |

The first two analyses are exploratory. The full-data FAISS setting changes a clustering default, and the prompt factorial was compared on CIFAR-10 test labels. Neither establishes an independently validated improvement. The original authors used FAISS GPU; these scripts use FAISS CPU.

To reproduce the completed jobs, run these commands in your own PowerShell terminal from `reproduction/`. On this machine, first set the existing local CLIP and FAISS package paths as shown in [Environment and cache](EXPERIMENTS.md#environment-and-cache); in a fresh environment, install `requirements-faiss.txt`.

```powershell
python run_ablation_grid.py --threads 8 --seeds 0 1 2 3 4 --restarts 20
python transfer_dataset.py cifar20 --prompt-set seven --backend faiss-cpu --threads 8
python transfer_dataset.py stl10 --prompt-set seven --backend faiss-cpu --threads 8
python train_tac.py --prompt-set seven --device cpu --threads 8 --epochs 20 --seeds 0 1 2 3 4
```

These commands are independent; start them one at a time. Each script reuses its completed caches or checkpoints when restarted with the same settings. See the [runbook](EXPERIMENTS.md) for output paths, optional controls, and detailed resume behavior.

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

The reproduction follows the [authors' public implementation](https://github.com/XLearning-SCU/2024-ICML-TAC). Local copies of the authors' original modules were not uploaded to this repository. The published scripts are standalone. The original training code requires CUDA and FAISS GPU; this repository's completed training adaptation uses FAISS CPU neighbors and PyTorch CPU. Its 20-seed mean remains below the paper's TAC ACC of 91.9 and includes two poor seeds. Datasets, model weights, and paper PDFs are not distributed here.
