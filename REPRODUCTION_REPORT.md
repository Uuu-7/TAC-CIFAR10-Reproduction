# Reproducing Text-Guided Image Clustering: A CPU Study of TAC with Stability Analysis and Cross-Dataset Validation

**Student:** Xia Yuqi  
**Article type:** Reproduction and experimental validation report  
**Report date:** 20 November 2026  
**Code and experimental records:** [TAC-CIFAR10-Reproduction](https://github.com/Uuu-7/TAC-CIFAR10-Reproduction)  
**Source study:** Li et al., *Image Clustering with External Guidance*, ICML 2024 [1]

## Abstract

**Objective:** This study reproduces the training-free and trained variants of TAC and investigates how implementation choices, stochastic optimization, and text-space construction affect image clustering. **Methods:** Frozen OpenAI CLIP ViT-B/32 features were combined with text representations retrieved from a filtered WordNet vocabulary. CIFAR-10 experiments included a scikit-learn Euclidean K-means approximation, FAISS CPU spherical K-means, single- and seven-prompt encodings, a four-cell prompt factorial, sampling diagnostics, and 12 parameter ablations. Full TAC was trained for 20 epochs under 20 random seeds. Two modifications, text weight 0.75 and a maximum of 20 nouns per semantic center, were evaluated on CIFAR-20 and STL-10. **Results:** The initial training-free approximation achieved 89.18% clustering accuracy (ACC), compared with the source paper's 90.10%. The trained CPU adaptation achieved ACC 90.81 ± 2.76%, normalized mutual information (NMI) 82.61 ± 1.49%, and adjusted Rand index (ARI) 81.84 ± 2.96% across 20 seeds. Two seeds yielded substantially lower accuracy than the remaining 18. In a matched training-free control, image-only ACC was 75.13%, versus 88.84% with retrieved text. Increasing the noun limit from 5 to 20 increased all three mean metrics on both transfer datasets at both evaluated text weights. **Conclusion:** The implementation recovers much of the reported TAC performance and identifies sampling and training variability as material reproducibility concerns. A larger selected vocabulary shows positive transfer in the evaluated settings, while the evidence for reducing text weight is mixed across datasets. Differences from the authors' GPU environment and the exploratory use of CIFAR-10 test scores limit claims of exact replication or general improvement.

**Keywords:** reproducibility; image clustering; TAC; CLIP; WordNet; spherical K-means; cross-modal distillation

## 1. Introduction

Image clustering seeks to organize images without using class labels for optimization. TAC introduces external textual guidance into this process by constructing a discriminative text vocabulary and retrieving text counterparts for image features [1]. Its training-free variant clusters concatenated image and text representations. Its trained variant learns image and text cluster heads through cross-modal neighbor distillation and regularization.

Reproducing TAC requires more than obtaining one score near a published table entry. Prompt construction, vocabulary selection, K-means geometry, training subsampling, and initialization can each affect the result. CPU adaptations also introduce differences from the authors' released GPU workflow [2]. Accordingly, this report addresses four questions:

1. How closely can a CPU implementation recover the published CIFAR-10 results?
2. How do clustering backend, sampling, and prompt placement affect observed performance?
3. What contribution does retrieved text make under matched clustering settings?
4. Do modifications explored on CIFAR-10 transfer to other datasets?

The repository contains scripts, seed-level CSV files, configuration records, and English execution instructions. This report describes completed experiments; optional unrun controls are identified explicitly.

## 2. Materials and Methods

### 2.1. Datasets and representation extraction

CIFAR-10 uses 50,000 training images to construct the text space and 10,000 test images for clustering evaluation. CIFAR-20 uses the official 20 coarse labels of CIFAR-100, with 50,000 training and 10,000 test images. STL-10 uses its 5,000-image training split and 8,000-image test split. The STL-10 unlabeled split is not used in this implementation.

OpenAI CLIP ViT-B/32 produces 512-dimensional image and text features. CLIP is frozen throughout these experiments. Features are normalized to unit length before cosine-based operations. Train images, without labels, determine semantic centers and noun selection. Test labels are used to calculate evaluation metrics; they are not inputs to feature construction or training losses.

Some CIFAR-10 parameter choices were compared using test metrics and are therefore exploratory. Subsequent transfer experiments hold those choices fixed rather than searching on the target dataset scores.

### 2.2. Discriminative vocabulary and text retrieval

Training image features are clustered into semantic centers: 167 for CIFAR-10 and CIFAR-20, and 17 for STL-10. After assigning images, cluster means are recomputed and normalized. Each WordNet noun is compared with the centers. A softmax across centers produces a center assignment and confidence for each noun. Up to gamma nouns with the largest confidence are retained per center; gamma is 5 unless specified otherwise. This rule can select fewer than the nominal center-count times gamma nouns.

For a unit image feature x and selected unit noun features t_j, soft retrieval is defined by

$$
a_j(x)=\frac{\exp(x^\top t_j/\tau)}{\sum_l\exp(x^\top t_l/\tau)},\qquad
r(x)=\operatorname{normalize}\left(\sum_j a_j(x)t_j\right),
$$

with retrieval temperature tau = 0.005 by default. Training-free clustering uses the concatenated representation

$$
z(x)=[x;\alpha r(x)],
$$

where alpha is the text weight. The baseline uses alpha = 1.0. The exploratory weight comparison includes alpha = 0.75. Hard retrieval replaces the weighted mean with the most similar noun feature.

### 2.3. Prompt conditions

The single-prompt condition uses `a photo of a {}.`. The seven-prompt condition averages noun features from the templates in the authors' released code: `itap of a {}.`, `a bad photo of the {}.`, `a origami {}.`, `a photo of the large {}.`, `a {} in a video game.`, `art of the {}.`, and `a photo of the small {}.`.

The seven-prompt cache uses a text-padding optimization that omits positions after the end-of-text token. A previously recorded 256-noun numerical check found a maximum absolute feature difference of 2.4 × 10^-6 from standard encoding. Separate cache names distinguish the optimized path.

The prompt factorial separates vocabulary identity from retrieval encoding. A noun set selected using one prompt condition is encoded using either condition, producing four cells. All cells preserve the same underlying WordNet noun order.

### 2.4. Clustering protocols

The initial implementation uses scikit-learn CPU Euclidean K-means as an approximation. Later experiments use FAISS CPU spherical K-means, with up to 300 iterations and 20 restarts per evaluation seed. Semantic-center fitting uses seed 0 and 10 restarts in the standard follow-up protocol. Unless stated otherwise, evaluation uses seeds 0–4.

Local FAISS 1.16.0 defaults to a maximum of 256 training points per centroid. For 10 clusters and 10,000 test images, this limits center training to 2,560 vectors, although assignments and metrics cover the full test set. The sampling diagnostic increases the cap to 1,000, allowing all 10,000 vectors to train the evaluation centers. This is a deliberate protocol variant, not an assertion that the authors used that cap.

The 12-configuration CIFAR-10 ablation grid and its image-only control use this full-data cap. Transfer experiments retain the original transfer script's default FAISS sampling behavior. Absolute scores from these different protocols should not be treated as one common controlled comparison.

### 2.5. Full TAC training

Two independent cluster heads map normalized 512-dimensional features through a linear layer, batch normalization, ReLU, a second linear layer, and a 10-class softmax. Training uses retrieved text for the 50,000 training images and exact FAISS CPU nearest-neighbor lists in both modalities. One neighbor per modality is sampled for each anchor during optimization.

The objective combines bidirectional cluster distillation, image/text prediction consistency, and an entropy-based balance term:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{distill}}+
\mathcal{L}_{\mathrm{consistency}}-5\left[H(\bar p_{\mathrm{text}})+H(\bar p_{\mathrm{image}})\right].
$$

The distillation temperature is 0.5. Adam uses learning rate 0.001 and betas (0.9, 0.99); batch size is 512, nearest-neighbor count is 50, and training lasts 20 epochs per seed. Incomplete final batches are dropped. Final predictions come from the image head. Evaluation is performed after the final epoch rather than selecting the best epoch using test labels.

The initial five seeds were extended to seeds 0–19. Completed checkpoints preserve seeds 0–4. This extension measures training randomness conditional on a fixed feature cache, selected vocabulary, and retrieved-text cache; it does not repeat the entire noun-selection pipeline independently for every seed.

### 2.6. Evaluation and environment

ACC is calculated after optimal Hungarian matching between cluster assignments and class labels. NMI and ARI are also reported. All scores are percentages. Unless otherwise stated, uncertainty is the sample standard deviation across seeds, not a confidence interval. Mean differences are expressed in percentage points. No statistical significance tests are reported.

Later experiments record Python 3.13.14, PyTorch 2.14.0+cpu, and FAISS 1.16.0 on Windows. The earlier proxy run records Python 3.12.14 and scikit-learn 1.9.0. Thus, the study includes both backend and environment differences. The authors' implementation uses GPU operations; exact cross-platform numerical equivalence has not been established. Cached and resumed invocation times are not interpreted as first-run end-to-end runtime benchmarks.

## 3. Results

### 3.1. Initial training-free reproduction

**Table 1. Published reference scores and the initial CPU Euclidean approximation.** Reproduced values are five-seed means; published values are the source table entries and have no uncertainty estimate here.

| Method | ACC | NMI | ARI |
|---|---:|---:|---:|
| CLIP K-means, source paper | 74.20 | 70.30 | 61.60 |
| CLIP K-means, CPU approximation | 75.15 | 71.30 | 62.57 |
| TAC without training, source paper | 90.10 | 80.80 | 79.80 |
| TAC without training, CPU approximation | 89.18 | 79.67 | 78.05 |

The approximation falls 0.92 ACC points below the reported training-free TAC result. Its retrieved-text representation improves ACC by 14.03 points over its own image baseline. However, proximity to the paper does not establish exact reproduction because the clustering geometry differs.

### 3.2. FAISS sampling and stability

**Table 2. Spherical K-means results under two evaluation sampling caps.** Different seed counts are stated explicitly.

| Sampling condition | Prompt | Seeds | ACC | NMI | ARI |
|---|---|---:|---:|---:|---:|
| Default cap | Single | 20 | 85.00 ± 5.59 | 78.40 ± 1.71 | 74.66 ± 3.95 |
| Default cap | Seven | 20 | 85.98 ± 5.78 | 79.74 ± 1.61 | 76.53 ± 4.31 |
| Full-data cap | Single | 5 | 89.11 ± 0.01 | 79.75 ± 0.02 | 77.75 ± 0.02 |
| Full-data cap | Seven | 5 | 88.84 ± 3.04 | 80.31 ± 1.60 | 78.40 ± 3.42 |

Under the default cap, 7 of 20 seeds in each prompt condition have ACC below 85%. Under the full-data cap, no single-prompt seed and one seven-prompt seed falls below that threshold among five runs. This supports sensitivity to sampling but does not show that the cap alone removes all instability. The threshold is descriptive and was not a prespecified failure criterion.

### 3.3. Prompt selection–encoding factorial

**Table 3. FAISS CPU prompt factorial, default cap, five seeds, 20 restarts.**

| Noun-selection prompt | Retrieval-encoding prompt | ACC | NMI | ARI |
|---|---|---:|---:|---:|
| Single | Single | 84.16 ± 6.79 | 78.35 ± 2.03 | 74.28 ± 4.87 |
| Single | Seven | 87.88 ± 5.26 | 80.34 ± 1.39 | 78.22 ± 3.88 |
| Seven | Single | 85.22 ± 5.73 | 78.96 ± 1.60 | 75.47 ± 3.95 |
| Seven | Seven | 83.67 ± 6.42 | 78.86 ± 2.10 | 74.52 ± 4.99 |

Single-prompt selection retains 776 nouns; seven-prompt selection retains 749. The largest mean ACC occurs with single-prompt selection and seven-prompt retrieval encoding. Large variation and test-set exploration preclude a confirmed superiority claim. The diagonal cells are the same five-seed conditions as the original FAISS baselines, rather than additional independent replications.

### 3.4. Component and parameter ablations

**Table 4. CIFAR-10 training-free ablations, seven prompts, full-data evaluation cap, five seeds.** Each row changes one factor from the baseline (167 centers, gamma 5, tau 0.005, soft retrieval). Image-only is a separate matched control.

| Configuration | ACC | NMI | ARI |
|---|---:|---:|---:|
| Image-only control | 75.13 ± 0.03 | 71.19 ± 0.02 | 62.53 ± 0.02 |
| Baseline | 88.84 ± 3.04 | 80.31 ± 1.60 | 78.40 ± 3.42 |
| 50 centers | 85.29 ± 5.78 | 79.62 ± 1.61 | 75.85 ± 4.09 |
| 100 centers | 88.74 ± 3.28 | 80.11 ± 1.51 | 78.46 ± 3.35 |
| 300 centers | 87.37 ± 6.35 | 80.35 ± 1.99 | 77.63 ± 5.05 |
| Gamma 1 | 87.98 ± 4.86 | 80.80 ± 1.22 | 78.43 ± 3.40 |
| Gamma 3 | 87.55 ± 6.37 | 80.55 ± 1.84 | 78.16 ± 4.80 |
| Gamma 10 | 87.47 ± 6.46 | 80.36 ± 1.81 | 78.01 ± 4.99 |
| Gamma 20 | 90.21 ± 0.04 | 80.82 ± 0.04 | 79.86 ± 0.08 |
| Tau 0.002 | 87.96 ± 3.01 | 78.87 ± 1.49 | 76.77 ± 3.18 |
| Tau 0.010 | 89.81 ± 0.15 | 81.32 ± 0.14 | 79.02 ± 0.25 |
| Tau 0.020 | 78.28 ± 0.04 | 76.14 ± 0.01 | 66.80 ± 0.08 |
| Hard retrieval | 87.04 ± 2.28 | 77.75 ± 0.68 | 75.06 ± 2.07 |

Retrieved text raises baseline mean ACC by 13.71 points relative to image-only clustering. Gamma 20 raises ACC by 1.36 points relative to the ablation baseline and has low observed variation across the five seeds. Tau 0.010 yields the largest mean NMI in the grid, whereas tau 0.020 substantially reduces performance. These observations motivated fixed-parameter transfer evaluation rather than additional target-dataset tuning.

### 3.5. Transfer of text weighting and vocabulary size

An earlier CIFAR-10 fusion-weight experiment found ACC/NMI/ARI of 89.29/80.10/78.17 at text weight 0.75, versus 89.18/79.67/78.05 at weight 1.0. That experiment used the Euclidean approximation. Weight 0.75 was subsequently held fixed for target-dataset comparison under the transfer FAISS protocol; this changes the backend context of the original observation.

**Table 5. Training-free transfer results, five seeds per condition.** Gamma 5 and 20 are compared within each weight. All entries are mean ± sample standard deviation.

| Dataset | Gamma | Text weight | ACC | NMI | ARI |
|---|---:|---:|---:|---:|---:|
| CIFAR-20 | 5 | 1.00 | 54.39 ± 1.86 | 59.58 ± 0.69 | 40.74 ± 2.24 |
| CIFAR-20 | 5 | 0.75 | 55.04 ± 3.23 | 59.07 ± 1.34 | 40.74 ± 2.61 |
| CIFAR-20 | 20 | 1.00 | 56.68 ± 2.94 | 60.40 ± 0.98 | 42.74 ± 1.26 |
| CIFAR-20 | 20 | 0.75 | 56.63 ± 2.27 | 59.18 ± 0.40 | 42.23 ± 0.73 |
| STL-10 | 5 | 1.00 | 97.43 ± 0.02 | 94.12 ± 0.03 | 94.41 ± 0.04 |
| STL-10 | 5 | 0.75 | 98.00 ± 0.06 | 95.08 ± 0.13 | 95.63 ± 0.14 |
| STL-10 | 20 | 1.00 | 98.28 ± 0.02 | 95.65 ± 0.04 | 96.23 ± 0.04 |
| STL-10 | 20 | 0.75 | 98.28 ± 0.02 | 95.73 ± 0.05 | 96.24 ± 0.05 |

At gamma 5, reducing text weight increases all three mean metrics on STL-10 but lowers NMI on CIFAR-20. Evidence for this weighting modification is therefore dataset dependent.

Increasing gamma from 5 to 20 increases all three means on both datasets at both weights. At weight 1.0, CIFAR-20 ACC rises by 2.29 points and STL-10 ACC by 0.85 points. The retained vocabulary grows from 815 to 3,129 nouns on CIFAR-20 and from 85 to 340 on STL-10. These results support the tested larger-vocabulary choice in these settings. They do not establish that 20 is globally optimal, that improvements are statistically significant, or that more nouns always help.

### 3.6. Full TAC training and seed variability

**Table 6. Trained TAC on CIFAR-10.** The five-seed result is a subset of the 20-seed extension, not an independent cohort.

| Condition | ACC | NMI | ARI |
|---|---:|---:|---:|
| TAC, source paper | 91.90 | 83.30 | 83.10 |
| CPU adaptation, seeds 0–4 | 90.05 ± 3.79 | 82.24 ± 2.02 | 81.06 ± 4.06 |
| CPU adaptation, seeds 0–19 | 90.81 ± 2.76 | 82.61 ± 1.49 | 81.84 ± 2.96 |

The 20-seed mean falls 1.09 ACC, 0.69 NMI, and 1.26 ARI points below the paper. Eighteen seeds yield ACC between 91.30% and 91.89%. Seeds 1 and 10 yield 83.28% and 82.23%, respectively. Both are retained in every aggregate. Thus, most runs approach the published score, but selecting only a favorable run would conceal material training variability.

## 4. Discussion

The experiments distinguish three findings. First, external text contributes substantial clustering information under the matched CIFAR-10 control. Second, implementation details can strongly affect apparent reproducibility: changing K-means geometry or sampling can alter both the mean and variation. Third, training commonly reaches near-paper performance but exhibits a lower-performing mode under some initializations.

The gamma=20 result is the strongest tested improvement candidate because its choice originated in CIFAR-10 exploration and was then fixed for two additional datasets. Its interpretation remains limited to the evaluated frozen features, prompt ensemble, center construction, and FAISS settings. The smaller-text-weight result is less consistent, illustrating why an improvement on one dataset should not be presented as a universal gain.

The observed poorer training seeds do not by themselves identify a mechanism. Class merging, neighbor structure, initialization, and optimization dynamics are possible explanations, but this study has not isolated their causes. Cluster-size diagnostics, class-aligned confusion matrices, and loss-component ablations would be appropriate future analyses.

## 5. Limitations

1. **Protocol equivalence:** FAISS CPU and a Windows CPU training path differ from the authors' GPU implementation. The Euclidean proxy and full-data sampling variant are explicitly separate protocols.
2. **Exploratory evaluation:** Prompt combinations, gamma, temperature, and text weights were explored using CIFAR-10 test metrics. Their CIFAR-10 gains are not independent confirmatory results.
3. **Scope of randomness:** Evaluation seeds vary cluster fitting or head training while most representation and noun-selection caches remain fixed. Reported variation does not cover every source of end-to-end uncertainty.
4. **Transfer coverage:** Only two additional datasets and five seeds per transfer condition were used. No significance tests or correction for multiple exploratory comparisons were performed.
5. **Incomplete mechanism analysis:** Full-WordNet retrieval and the scikit-learn noun-selection factorial extension remain unrun; trained loss-component ablations and poorer-seed diagnosis are also outside the completed experiments.
6. **Runtime and artifacts:** Saved invocation times can exclude earlier cached or interrupted work. Large embeddings, datasets, model weights, and checkpoints are not distributed in the repository.

## 6. Conclusion

This reproduction implements and evaluates both training-free TAC and full cross-modal distillation. It recovers much of the published CIFAR-10 performance while exposing sensitivity to clustering sampling and training seeds. The final trained CPU result is ACC 90.81 ± 2.76%, NMI 82.61 ± 1.49%, and ARI 81.84 ± 2.96% across 20 seeds. A matched control supports the contribution of retrieved text, and fixed gamma=20 transfer experiments improve all three mean metrics in the evaluated target conditions. These findings support a reproducible CPU adaptation with measured limitations rather than a claim of exact GPU replication.

## Data and Code Availability

The implementation, configuration files, and seed-level results are available at **[https://github.com/Uuu-7/TAC-CIFAR10-Reproduction](https://github.com/Uuu-7/TAC-CIFAR10-Reproduction)**. The README provides the project overview, while [EXPERIMENTS.md](EXPERIMENTS.md) provides PowerShell commands and resume instructions. The datasets, CLIP weights, source paper PDFs, and generated feature caches are excluded from version control. Local copies of the authors' original modules are also excluded; readers should consult the upstream repository [2].

**Table-to-artifact mapping:**

| Report evidence | Repository artifact |
|---|---|
| Initial approximation | [results/summary.json](reproduction/results/summary.json) and [runs.csv](reproduction/results/runs.csv) |
| Sampling stability | [default-cap runs](reproduction/results/stability_default.csv) and [full-data runs](reproduction/results/stability_full_data.csv), each with a neighboring config JSON |
| Prompt factorial | [factorial_faiss.csv](reproduction/results/factorial_faiss.csv) and [configuration](reproduction/results/factorial_faiss.config.json) |
| Ablation grid | [ablation summary](reproduction/results/ablation_grid/summary.csv), individual CSV files, and config JSON files |
| Image-only control | [ablation_image_only.csv](reproduction/results/ablation_image_only.csv) and [configuration](reproduction/results/ablation_image_only.config.json) |
| CIFAR-10 fusion weight | [fusion_ablation.csv](reproduction/results/fusion_ablation.csv) |
| Gamma 5 transfer | [CIFAR-20 summary](reproduction/results/cifar20_transfer/summary.json) and [STL-10 summary](reproduction/results/stl10_transfer/summary.json), with adjacent runs CSV files |
| Gamma 20 transfer | [CIFAR-20 summary](reproduction/results/cifar20_transfer_gamma20/summary.json) and [STL-10 summary](reproduction/results/stl10_transfer_gamma20/summary.json), with adjacent runs CSV files |
| Full TAC training | [five-seed summary](reproduction/results/tac_training/summary.json), [20-seed summary](reproduction/results/tac_training_20seeds/summary.json), and [20-seed runs](reproduction/results/tac_training_20seeds/runs.csv) |

## References

[1] Li et al. *Image Clustering with External Guidance*. Proceedings of the 41st International Conference on Machine Learning, 2024. [Published paper and supplementary resources](https://proceedings.mlr.press/v235/li24aa.html).

[2] XLearning-SCU. *2024-ICML-TAC: Authors' implementation*. [GitHub repository](https://github.com/XLearning-SCU/2024-ICML-TAC).

[3] Uuu-7. *TAC-CIFAR10-Reproduction: CPU reproduction and experimental records*. [GitHub repository](https://github.com/Uuu-7/TAC-CIFAR10-Reproduction).
