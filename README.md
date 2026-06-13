# HEST-Benchmark Table 1 Reproduction

Reproducing the HEST-Benchmark (arXiv **2406.16192**) Table 1 gene-expression
prediction results: extract H&E patch features with **CONCH** and **Virchow2**,
train Ridge regression to predict spot-level expression of the top-50 highly
variable genes, and score with **Pearson correlation** — for **BRCA**, **CRC**,
and **LUAD**.

**Bonus:** integrate SEAL (`mahmoodlab/SEAL`) CONCH-SEAL / UNIv2-SEAL as custom
encoders in the Trident feature-extraction pipeline.

> **Status:** this repository is a *scaffold* built on a no-GPU Mac. The
> GPU-independent pieces (the ridge/Pearson evaluation, config loading, the
> encoder interface) are implemented and tested; the GPU-bound steps (real
> feature extraction, the full benchmark) are structured and ready to run once
> on a GPU Linux machine. See **Current status** at the bottom.

## Upstream projects
- HEST: https://github.com/mahmoodlab/HEST
- Trident: https://github.com/mahmoodlab/trident
- SEAL: https://github.com/mahmoodlab/SEAL

These are cloned into `third_party/` (git-ignored) purely as a source reference.

## How the benchmark works (what we reproduce)
- Per task, predict the **50 most highly variable genes** (`var_50genes.json`)
  from an H&E patch at each spatial-transcriptomics spot.
- Embeddings → **PCA(256)** (StandardScaler + PCA) → **Ridge regression**
  (`alpha = 100 / (n_features · n_genes)`, `solver='lsqr'`, `fit_intercept=False`).
- Score = per-gene **Pearson correlation**, averaged over genes and k-fold splits.
- Expression normalized per spot with total-count + `log1p`.

Cancer type → HEST task code: **BRCA → IDC**, **CRC → COAD/READ**, **LUAD → LUNG**.

## Repository layout
```
.
├── README.md / PROGRESS.md        # overview + two-week checklist
├── environment.yml                # conda env (CUDA notes inline)
├── .env.example                   # HF_TOKEN / DATA_DIR / RESULTS_DIR
├── configs/
│   ├── paths.yaml                 # the ONE place paths are defined (env-driven)
│   └── {brca,crc,luad}_{conch,virchow2}.yaml
├── src/
│   ├── encoders/conch_seal_encoder.py   # SEAL → Trident BasePatchEncoder wrapper
│   ├── eval/ridge_eval.py               # ridge + Pearson (tested on CPU)
│   └── utils/config.py                  # config + path resolution
├── scripts/
│   ├── setup_env.sh               # conda env + editable installs
│   ├── check_hf_access.py         # verify gated HF access (metadata only)
│   ├── extract_features.py        # Trident extraction (GPU)
│   └── run_benchmark.py           # synthetic (CPU) + real (GPU) benchmark
├── notes/                         # two-week plan, one file per day block
├── tests/                         # pytest: ridge eval + encoder interface
└── third_party/                   # HEST / Trident / SEAL clones (git-ignored)
```

## Two-week plan (overview)
| Days | Focus | Note |
|------|-------|------|
| 1–2 | Pathology AI review | `notes/day1-2_*` |
| 3–4 | HEST paper notes | `notes/day3-4_*` |
| 5–6 | Environment & data setup | `notes/day5-6_*` |
| 7 | Trident demo | `notes/day7_*` |
| 8–9 | Feature extraction (CONCH, Virchow2) | `notes/day8-9_*` |
| 10–11 | Ridge evaluation | `notes/day10-11_*` |
| 12 | Comparison with paper | `notes/day12_*` |
| 13–14 | SEAL bonus | `notes/day13-14_*` |

See `PROGRESS.md` for the live checklist.

## Setup
1. Clone the three upstream repos into `third_party/` (one-time):
   ```bash
   mkdir -p third_party && cd third_party
   git clone https://github.com/mahmoodlab/HEST.git
   git clone https://github.com/mahmoodlab/trident.git
   git clone https://github.com/mahmoodlab/SEAL.git
   cd ..
   ```
2. Create the env and install the repos (editable):
   ```bash
   bash scripts/setup_env.sh
   ```
   > On the **GPU machine**, install the CUDA build of torch *before* this — see
   > the note at the top of `environment.yml`.
3. Configure secrets/paths:
   ```bash
   cp .env.example .env      # then fill HF_TOKEN, DATA_DIR, RESULTS_DIR
   ```
4. Verify gated Hugging Face access (safe on a laptop, metadata only):
   ```bash
   python scripts/check_hf_access.py
   ```

## Usage
```bash
# CPU sanity check of the ridge + Pearson core (no GPU, no data):
python scripts/run_benchmark.py --synthetic

# Real benchmark on the GPU machine (per task):
python scripts/run_benchmark.py --config configs/brca_conch.yaml
python scripts/run_benchmark.py --config configs/luad_virchow2.yaml

# SEAL bonus encoder:
python scripts/run_benchmark.py --config configs/brca_conch.yaml --custom-encoder conch_seal

# Trident feature extraction from raw WSIs (GPU):
python scripts/extract_features.py --wsi-dir $WSI_DIR --job-dir out --encoder conch_v1

# Tests:
pytest tests/
```

Paths come from `configs/paths.yaml` (which reads `${DATA_DIR}` / `${RESULTS_DIR}`
from your environment), so switching machines only means editing `.env`.

## Current status (built on a no-GPU Mac)
**Verified to run on CPU here:**
- `pytest tests/` — 10 tests pass (ridge/Pearson math + SEAL encoder interface).
- `python scripts/run_benchmark.py --synthetic` — full ridge eval on synthetic data.
- `python scripts/check_hf_access.py` — runs and reports per-repo access.
- Config loading with env-var path expansion.

**Scaffolded, runs on the GPU machine only** (needs CUDA, the editable installs,
and gated HF access): `scripts/extract_features.py`, the real-mode
`scripts/run_benchmark.py --config ...`, and the SEAL `--custom-encoder` path.

### First things to do on the GPU machine
1. `git clone` this repo, re-clone the three upstream repos into `third_party/`
   (they are git-ignored), then `bash scripts/setup_env.sh` with the CUDA torch
   build installed first.
2. `cp .env.example .env`, fill in `HF_TOKEN` / `DATA_DIR` / `RESULTS_DIR`, then
   run `python scripts/check_hf_access.py` until **all repos show OK** (request
   gated access for CONCH, Virchow2, SEAL, hest-bench if needed).
3. Smoke-test one task: `python scripts/run_benchmark.py --config configs/brca_conch.yaml`
   (HEST auto-downloads the bench data on first run), confirm a Pearson number
   comes out, then fan out to the other 5 configs.

## Caveats / decisions to confirm with the mentor
- The mentor's `HEST_Benchmark_Task_Summary.md` (repo root) has been
  cross-checked against this scaffold: cancer types (BRCA/CRC/LUAD), encoders
  (CONCH + Virchow2), top-50 HVGs, 256px @ 20x, Ridge + Pearson, and the SEAL
  bonus all match. Note its section 八 reference magnitudes (CONCH BRCA
  ~0.20–0.28) are rough; the HEST README snapshot used in `notes/day10-11_*`
  reports higher numbers (BRCA/IDC CONCH 0.5363) — confirm which to compare
  against.
- "CONCH" defaults to Trident's **`conch_v1`** (the original Table 1 entry);
  `conch_v15` is also wired in if v1.5 is intended.
- CRC configs include **both COAD and READ**; trim to one if the mentor wants a
  single colorectal task.
- Patching uses **256px @ 20x** (per the setup prompt / Trident defaults); the
  official HEST data is tiled at 112µm @ 0.5µm/px.
