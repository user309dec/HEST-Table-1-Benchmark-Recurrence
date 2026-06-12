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

## Notes
_(your notes here)_
