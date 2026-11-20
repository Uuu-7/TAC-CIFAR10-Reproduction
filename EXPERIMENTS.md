# Additional TAC experiments

These commands are intended for **your own PowerShell terminal**. The user has completed the default 12-configuration ablation grid, CIFAR-20/STL-10 transfer, and five-seed TAC training. Optional WordNet-wide retrieval and the scikit-learn selection factorial extension remain unrun. All scripts write outputs under `reproduction/results/`. Run commands from `reproduction/`.

## Environment and cache

The already completed CIFAR-10 runs use Python 3.13, PyTorch CPU, FAISS CPU 1.16.0, and cached OpenAI CLIP ViT-B/32 features. The authors' implementation uses FAISS GPU; FAISS CPU is a protocol difference. `pip install -r requirements-faiss.txt` installs the declared dependencies in a fresh environment. On the existing machine, the locally installed CLIP and FAISS packages are in `.deps-clip` and `.deps-faiss`. The following setup uses those installed packages:

```powershell
cd 'E:\Data Science\DS_2\2_Research Seminar\reproduction'
$env:PYTHONPATH='E:\Data Science\DS_2\2_Research Seminar\.deps-clip;E:\Data Science\DS_2\2_Research Seminar\.deps-faiss'
python -c "import clip, faiss, torch; print('FAISS', faiss.__version__, 'Torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

Keep the same `PYTHONPATH` in every new PowerShell session. On a new environment with the correct packages installed normally, it is unnecessary. The scripts reuse the existing files under `data/cache/`; large feature caches and checkpoints are ignored by Git. Do not change the WordNet noun order or CLIP model while reusing those caches.

## 1. FAISS stability and sampling diagnostic

Completed diagnostic files: [`stability_default.csv`](reproduction/results/stability_default.csv) (20 seeds for each prompt; FAISS default training sample cap) and [`stability_full_data.csv`](reproduction/results/stability_full_data.csv) (five seeds for each prompt; cap raised above the 10,000-image test set). Each has a neighboring `.config.json` with the exact arguments. The first diagnostic uses 20 restarts per seed, as in the previous FAISS runs. Local FAISS 1.16.0 defaults to at most 256 training points per centroid, so a 10-cluster run trains on at most 2,560 of 10,000 test vectors. Raising this to 1,000 includes all 10,000; it is a *diagnostic protocol variant*, not an exact replay of the authors' GPU environment.

| Setting | Prompt | Seeds | ACC mean ± SD | NMI mean ± SD | ARI mean ± SD |
|---|---|---:|---:|---:|---:|
| FAISS default cap | Single | 20 | 85.00 ± 5.59 | 78.40 ± 1.71 | 74.66 ± 3.95 |
| FAISS default cap | Seven | 20 | 85.98 ± 5.78 | 79.74 ± 1.61 | 76.53 ± 4.31 |
| Full test-set training | Single | 5 | 89.11 ± 0.01 | 79.75 ± 0.02 | 77.75 ± 0.02 |
| Full test-set training | Seven | 5 | 88.84 ± 3.04 | 80.31 ± 1.60 | 78.40 ± 3.42 |

Under the default cap, 7 of 20 seeds for each prompt had ACC below 85. Raising the cap removed this outcome for the single prompt in five seeds, but one of five seven-prompt seeds remained below 85. This suggests sampling affects stability; it does not establish a general fix. For additional seeds, rerun the following with a larger `--seeds` list and a new output filename:

```powershell
python cached_experiments.py stability --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --restarts 20 --threads 8 --max-points-per-centroid 1000 --output results/stability_full_data_20.csv
```

## 2. Separate noun selection from prompt encoding

[`factorial_faiss.csv`](reproduction/results/factorial_faiss.csv) contains the four selection/retrieval combinations with the same FAISS backend, seeds 0–4, and 20 restarts. The two diagonal cells reproduce the previous single- and seven-prompt FAISS results. Off-diagonal cells reuse the *noun identities* selected by one prompt and encode those same nouns with the other prompt. The WordNet noun order must therefore be identical across caches.

| Noun selection prompt | Retrieval encoding prompt | ACC mean ± SD | NMI mean ± SD | ARI mean ± SD |
|---|---|---:|---:|---:|
| Single | Single | 84.16 ± 6.79 | 78.35 ± 2.03 | 74.28 ± 4.87 |
| Single | Seven | 87.88 ± 5.26 | 80.34 ± 1.39 | 78.22 ± 3.88 |
| Seven | Single | 85.22 ± 5.73 | 78.96 ± 1.60 | 75.47 ± 3.95 |
| Seven | Seven | 83.67 ± 6.42 | 78.86 ± 2.10 | 74.52 ± 4.99 |

The single-selection/seven-encoding cell has the highest mean among these four *exploratory* cells. Its seed variation is still large. A gain claim requires independent validation or a selection rule fixed before looking at test labels.

For a fuller backend factorial, the following also tests scikit-learn noun selection. It refits 167 semantic centers and may take a long time on CPU:

```powershell
python cached_experiments.py factorial --selection-backends sklearn faiss-cpu --seeds 0 1 2 3 4 --restarts 20 --threads 8 --output results/factorial_all_backends.csv
```

## 3. Component and parameter ablations

All 12 default configurations are complete (five seeds each); see [summary.csv](reproduction/results/ablation_grid/summary.csv). Baseline ACC/NMI/ARI are 88.84/80.31/78.40. Gamma 20 gives 90.21/80.82/79.86, while temperature 0.01 gives 89.81/81.32/79.02. These test-set comparisons are exploratory. The completed [image-only control](reproduction/results/ablation_image_only.csv) gives ACC 75.13 ± 0.03, NMI 71.19 ± 0.02, ARI 62.53 ± 0.02 using identical seeds, restarts and evaluation sample cap. Equal-weight image/text fusion at baseline improves mean ACC by 13.71 percentage points. The all-noun control has not been run.

The prespecified grid varies one factor from the seven-prompt baseline: semantic centers (50/100/300), nouns per center (1/3/10/20), retrieval temperature (0.002/0.01/0.02), and hard vs soft retrieval. It uses FAISS CPU, five seeds, 20 restarts, and the full test-set training cap of 1,000. Each configuration writes its own CSV, config JSON, and log. Restarting skips completed configurations.

```powershell
python run_ablation_grid.py --threads 8 --seeds 0 1 2 3 4 --restarts 20
```

The image-only control is separate:

```powershell
python cached_experiments.py ablation --prompts seven --image-only --no-selected --max-points-per-centroid 1000 --seeds 0 1 2 3 4 --restarts 20 --output results/ablation_image_only.csv
```

The full-WordNet retrieval control is especially costly on CPU and is **off by default**. To include it, rerun the grid with `--include-all-nouns`; completed cells are skipped. To run only that control:

```powershell
python cached_experiments.py ablation --prompts seven --all-nouns --no-selected --max-points-per-centroid 1000 --seeds 0 1 2 3 4 --restarts 20 --output results/ablation_all_nouns.csv
```

Do not select a parameter using these CIFAR-10 test scores and then report its CIFAR-10 score as a confirmed improvement.

## 4. Transfer and independent validation

`transfer_dataset.py` runs the training-free pipeline on CIFAR-20 (official CIFAR-100 coarse labels; 20 clusters) and STL-10 (10 clusters). It uses 167/17 semantic centers, respectively, and reuses the cached seven-prompt WordNet embeddings. It compares equal text weight `1.0` with `0.75`, the exploratory weight observed on CIFAR-10, **without tuning on either target dataset**. Its image extraction is resumable after each batch; rerun the same command after an interruption.

```powershell
python transfer_dataset.py cifar20 --prompt-set seven --backend faiss-cpu --threads 8 --output-dir results/cifar20_transfer
python transfer_dataset.py stl10 --prompt-set seven --backend faiss-cpu --threads 8 --output-dir results/stl10_transfer
```

Both transfers are complete. Each output directory contains `runs.csv` and `summary.json`. Scores below are mean ± sample SD over seeds 0–4:

| Dataset | Text weight | ACC | NMI | ARI |
|---|---:|---:|---:|---:|
| CIFAR-20 | 1.0 | 54.39 ± 1.86 | 59.58 ± 0.69 | 40.74 ± 2.24 |
| CIFAR-20 | 0.75 | 55.04 ± 3.23 | 59.07 ± 1.34 | 40.74 ± 2.61 |
| STL-10 | 1.0 | 97.43 ± 0.02 | 94.12 ± 0.03 | 94.41 ± 0.04 |
| STL-10 | 0.75 | 98.00 ± 0.06 | 95.08 ± 0.13 | 95.63 ± 0.14 |

Weight 0.75 improves all three mean metrics on STL-10. On CIFAR-20, ACC increases slightly, NMI decreases, and ARI is nearly unchanged; this is mixed evidence for transfer.

## 5. Full TAC distillation

`train_tac.py` implements the released two-head architecture, 50 image/text nearest neighbors, bidirectional cluster distillation, consistency and balance losses, 20 epochs, batch size 512, and Adam learning rate 0.001. It constructs retrieved *training* text from the cached seven-prompt noun set, mines exact FAISS CPU neighbors, and saves a checkpoint after each epoch. Restarting the same command resumes completed epochs. Test labels are used only for the final score for each seed. This is a CPU adaptation of the authors' GPU training path.

```powershell
python train_tac.py --prompt-set seven --device cpu --threads 8 --epochs 20 --seeds 0 1 2 3 4 --output-dir results/tac_training
```

All five seeds completed 20 epochs. Results are in [runs.csv](reproduction/results/tac_training/runs.csv) and [summary.json](reproduction/results/tac_training/summary.json). ACC/NMI/ARI are **90.05 ± 3.79 / 82.24 ± 2.02 / 81.06 ± 4.06**, compared with the paper's 91.9/83.3/83.1. Seed ACCs are 91.83, 83.28, 91.63, 91.74, and 91.78; the poor seed is retained in every aggregate. The saved invocation reports 210 seconds, which is not a reliable first-run end-to-end cost because cached preparation or resumed work may precede it.

Checkpoints and neighbors stay in ignored `data/cache/`. Keep the same output directory, prompt, batch size, neighbor count, temperature, balance weight, and epoch target when resuming. If you change the noun cache or CLIP features, clear the corresponding retrieved-text, neighbor, and checkpoint caches first.

## Reading results

### Follow-up: transfer the gamma=20 choice

After the 20-seed training extension and image-only control, test the fixed CIFAR-10 choice of 20 nouns per center against the completed gamma=5 transfer runs. Keep prompts, seeds, FAISS settings, and both text weights unchanged. Compare gamma values within each weight; do not choose a new weight using target scores. Reuse cached image/WordNet features and write separate output folders:

```powershell
python transfer_dataset.py cifar20 --prompt-set seven --backend faiss-cpu --threads 8 --gamma 20 --seeds 0 1 2 3 4 --weights 1.0 0.75 --output-dir results/cifar20_transfer_gamma20
python transfer_dataset.py stl10 --prompt-set seven --backend faiss-cpu --threads 8 --gamma 20 --seeds 0 1 2 3 4 --weights 1.0 0.75 --output-dir results/stl10_transfer_gamma20
```

The gamma=20 follow-up is complete. See the CIFAR-20 [runs](reproduction/results/cifar20_transfer_gamma20/runs.csv) / [summary](reproduction/results/cifar20_transfer_gamma20/summary.json) and STL-10 [runs](reproduction/results/stl10_transfer_gamma20/runs.csv) / [summary](reproduction/results/stl10_transfer_gamma20/summary.json). For weights 1.0/0.75, CIFAR-20 ACC is 56.68 ± 2.94 / 56.63 ± 2.27; STL-10 ACC is 98.28 ± 0.02 / 98.28 ± 0.02. All three mean metrics increase relative to gamma=5 within each weight. Five paired seeds per dataset provide evidence of transfer for this fixed choice, but no significance test or universal improvement claim is made. The selected vocabulary grows from 815 to 3,129 nouns on CIFAR-20 and from 85 to 340 on STL-10.

### Follow-up: full TAC with 20 seeds

The completed [20-seed extension](reproduction/results/tac_training_20seeds/summary.json) uses seeds 0–19, preserving the original seeds 0–4 via completed checkpoints. ACC/NMI/ARI are 90.81 ± 2.76 / 82.61 ± 1.49 / 81.84 ± 2.96. Seeds 1 and 10 have ACC 83.28 and 82.23; the other 18 range from 91.30 to 91.89. All seeds remain in the summary. This measures training randomness conditional on one fixed CLIP feature and selected-text cache, rather than randomness of the entire feature/noun-selection pipeline.

```powershell
python train_tac.py --prompt-set seven --device cpu --threads 8 --epochs 20 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --output-dir results/tac_training_20seeds
```

Scores are percentages. Means and standard deviations summarize the requested random seeds; a low-variance result on five seeds does not prove stability across environments. All ablations evaluate on the test set for analysis, while feature construction and noun selection use train images without labels. The paper's TAC (no train) CIFAR-10 row is ACC 90.1, NMI 80.8, ARI 79.8; the trained TAC row is ACC 91.9, NMI 83.3, ARI 83.1. See the [paper](https://proceedings.mlr.press/v235/li24aa.html) and [released code](https://github.com/XLearning-SCU/2024-ICML-TAC) for the original protocol.
