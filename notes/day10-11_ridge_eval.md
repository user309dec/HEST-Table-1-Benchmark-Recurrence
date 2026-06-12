# Day 10–11 — Ridge evaluation

## Goal
Run the ridge + Pearson evaluation on the extracted embeddings and produce
per-task, per-encoder Pearson scores.

## The math (implemented in `src/eval/ridge_eval.py`, verified on synthetic data)
- PCA(256) with `StandardScaler` fit on train, applied to test.
- `alpha = 100 / (n_features * n_genes)`.
- `Ridge(solver='lsqr', alpha=alpha, fit_intercept=False, random_state=seed, max_iter=1000)`.
- Per-gene Pearson → mean over genes → mean over k folds.

## Run
```bash
# CPU sanity check (no GPU/data):
python scripts/run_benchmark.py --synthetic

# Real run per task:
python scripts/run_benchmark.py --config configs/brca_conch.yaml
python scripts/run_benchmark.py --config configs/brca_virchow2.yaml
# ... crc_*, luad_*
```

## Results (fill in as runs complete)
| Task | Encoder | Pearson mean | Pearson std | HEST README ref |
|------|---------|--------------|-------------|-----------------|
| BRCA (IDC) | CONCH v1   |  |  | 0.5363 |
| BRCA (IDC) | Virchow2   |  |  | 0.5971 |
| CRC (COAD) | CONCH v1   |  |  | 0.2489 |
| CRC (COAD) | Virchow2   |  |  | 0.2581 |
| LUAD (LUNG)| CONCH v1   |  |  | 0.5322 |
| LUAD (LUNG)| Virchow2   |  |  | 0.5685 |

(Reference numbers are from the HEST README results table, 03.04.26 snapshot —
they may differ slightly from the original paper's Table 1.)

## Notes
_(your notes here)_
