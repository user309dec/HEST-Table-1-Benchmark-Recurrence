# Day 5–6 — Environment & data setup

## Goal
Stand up the conda env, install HEST/Trident/SEAL, confirm gated HF access, and
(on the GPU box) pre-download the benchmark data.

## Checklist
- [ ] `conda env create -f environment.yml` (or `bash scripts/setup_env.sh`).
- [ ] On GPU machine: install CUDA torch build first (see environment.yml note).
- [ ] `bash scripts/setup_env.sh` → editable installs of Trident, HEST, SEAL.
- [ ] Copy `.env.example` → `.env`; fill `HF_TOKEN`, `DATA_DIR`, `RESULTS_DIR`.
- [ ] `python scripts/check_hf_access.py` → all repos OK.
- [ ] HEST downloads bench data automatically on first `benchmark()` run
      (`MahmoodLab/hest-bench`), or pre-fetch it.

## Gated Hugging Face repos needed
- `MahmoodLab/conch`, `MahmoodLab/conchv1_5` — CONCH encoders
- `paige-ai/Virchow2` — Virchow2 encoder
- `MahmoodLab/SEAL` — SEAL checkpoints (bonus)
- `MahmoodLab/hest-bench` (dataset) — eval data
- `MahmoodLab/hest` (dataset) — full HEST-1k (optional)

## Notes
_(your notes here — record which access requests are approved and when)_
