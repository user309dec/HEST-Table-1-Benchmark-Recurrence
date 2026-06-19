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

## Current status
**Verified to run on CPU here:**
- `pytest tests/` — 10 tests pass (ridge/Pearson math + SEAL encoder interface).
- `python scripts/run_benchmark.py --synthetic` — full ridge eval on synthetic data.
- `python scripts/check_hf_access.py` — runs and reports per-repo access.
- Config loading with env-var path expansion.

**runs on the GPU machine only**: `scripts/extract_features.py`, the real-mode
`scripts/run_benchmark.py --config ...`, and the SEAL `--custom-encoder` path.

## Results 
| Task | conch_v1 | virchow2 |
|------|----------|----------|
| BRCA / IDC | **0.5363** | **0.5971** |
| CRC (COAD/READ avg) | **0.2044** (COAD .249 / READ .160) | **0.2326** (COAD .258 / READ .207) |
| LUAD / LUNG | **0.5323** | **0.5685** |

Virchow2 > CONCH on every task (as in the paper). BRCA/IDC conch_v1 = 0.5363 matches the HEST README value exactly.

### SEAL bonus (CONCH SEAL)
| Task | CONCH | CONCH-SEAL | Δ | Virchow2 |
|------|-------|------------|---|----------|
| BRCA / IDC | 0.5363 | **0.5484** | **+0.012** | 0.5971 |
| CRC (avg)  | 0.2044 | **0.2245** | **+0.020** | 0.2326 |
| LUAD / LUNG | 0.5323 | **0.5349** | **+0.003** | 0.5685 |

CONCH-SEAL (LoRA-fine-tuned CONCH ViT-B/16) beats vanilla CONCH on every task —
SEAL's ST-molecular fine-tuning helps, largest gain on CRC. Still below the much
larger Virchow2 ViT-H.

How it was made to run despite SEAL pinning transformers 4.48 + timm 1.0.9 (conflicts with the main stack's 4.40.2 / 0.9.16):
- Did NOT install SEAL's full deps. Added only minimal vision-path extras: `peft==0.11.1`, `accelerate==0.30.1` (`--no-deps`), and `setuptools<81` (restores `pkg_resources`). Kept `huggingface_hub` at 0.36.2 (<1.0, required by transformers 4.40.2; also ships the `hf` CLI SEAL needs).
- Ran the benchmark with **CWD = `third_party/SEAL`** (so SEAL's relative `conf/config.yaml` + `weights/` resolve) and **`.venv/bin` on PATH** (so SEAL's `shutil.which('hf')` download path works). Checkpoints `seal_conch_{vision,omics}.pth` auto-download to `weights/conch_SEAL/`. The CONCH-SEAL vision encoder is a LoRA-adapted CONCH ViT-B/16.
- Helper: `logs/_run_seal3.sh`.

