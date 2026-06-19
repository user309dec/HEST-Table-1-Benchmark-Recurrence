# Day 3–4 — HEST paper notes

> Paper: HEST-1k / HEST-Benchmark, arXiv 2406.16192.

## Goal
Understand the HEST-Benchmark task and the exact Table 1 evaluation protocol we
are reproducing.

## What the benchmark does (from the cloned source)
- Per task: predict the **50 most highly variable genes** (shipped as
  `var_50genes.json`) from an H&E patch centered on each ST spot.
- Patches: 112×112 µm regions at 0.5 µm/px in the official data; our Trident
  extraction uses **256 px @ 20x** (matches the Trident tutorial defaults).
- Model = **Ridge regression** with PCA(256) dimensionality reduction on the
  patch embeddings (PCA makes encoders of different dim comparable).
- Metric = **Pearson correlation** per gene, averaged over genes, then over
  k-fold splits.
- Ridge hyper-params (from `hest/bench/trainer.py`):
  `alpha = 100 / (n_features * n_genes)`, `solver='lsqr'`,
  `fit_intercept=False`, `max_iter=1000`.
- Expression normalization: per-spot total count + `log1p` (`sc.pp.log1p`).

## Cancer type → HEST task code
| This project | HEST task code(s) | Organ |
|---|---|---|
| BRCA | IDC (and LYMPH_IDC) | breast |
| CRC  | COAD, READ | colon / rectum |
| LUAD | LUNG | lung |

## The dataset (HEST-1k), web-verified 2026-06-18
- **1,229 spatial-transcriptomic profiles**, each paired with a WSI + metadata.
- Assembled from **153 cohorts**, **26 organs**, **2 species** (human + mouse),
  **367 cancer samples across 25 cancer types**.
- Yields **~2.1M expression–morphology pairs** and **>76M segmented nuclei**.
- Published at **NeurIPS 2024 (Spotlight)** by the Mahmood Lab; ships
  **HEST-Library** (data processing) and **HEST-Benchmark** (this evaluation).
- We use the curated **`MahmoodLab/hest-bench`** subset (per-task patches +
  `*.h5ad` expression + `splits/` + `var_50genes.json`), not the full 2 TB set.

## How the benchmark ranks models (the bigger picture)
- **9 organ/cancer tasks**; **25 public patch encoders** evaluated.
- Leaderboard ordering (avg Pearson): top is **H-Optimus-1 (~0.423)** ≈
  GenBio-PathFM ≈ H-Optimus-0; **Virchow2 / UNI2 / CONCH v1.5** in the first
  tier; **ResNet-50 baseline ~0.325** at the bottom — i.e. pathology SSL clearly
  beats ImageNet pretraining. (We reproduce a *subset*, not the whole table.)

## The evaluation, restated as pseudocode
```
for task in {IDC, COAD, READ, LUNG}:
    for each ST spot: patch = crop H&E (256px@20x); x = encoder(patch)  # [N, D]
    y = log1p(total_count_normalize(expression[:, top50_HVG]))          # [N, 50]
    for fold in official_kfold_splits:
        Xtr,Xte = PCA256(StandardScaler(x))         # fit on train only
        w = Ridge(alpha=100/(D*50), lsqr, no_intercept).fit(Xtr, ytr)
        score_fold = mean_g Pearson(yte[:,g], (Xte*w)[:,g])
    report mean/std over folds
```

## Notes / takeaways
- The metric is **Pearson, not error** — we care whether morphology can *rank*
  high vs low expression, which is scale-invariant. Hence the very negative R²
  values seen in logs are expected and not the headline number.
- PCA(256) is the fairness knob: it stops high-dim encoders (Virchow2 2560-d)
  from winning just by having more regression degrees of freedom.
- Protocol nuance we accept: official tiles are 112 µm @ 0.5 µm/px; our configs
  use 256 px @ 20x — close but not identical. Despite this, our BRCA/IDC CONCH
  reproduced **0.5363 exactly** (see `day12_comparison.md`).

## Sources
- Jaume, Doucet, et al., "HEST-1k: A Dataset for Spatial Transcriptomics and
  Histology Image Analysis", **arXiv:2406.16192**, NeurIPS 2024 Spotlight.
- HEST repo + leaderboard: github.com/mahmoodlab/HEST;
  dataset: hf.co/datasets/MahmoodLab/hest-bench.
