# Day 5–6 — Environment & data setup

## Goal
Stand up the conda env, install HEST/Trident/SEAL, confirm gated HF access, and
(on the GPU box) pre-download the benchmark data.

## Checklist
- [ ] `conda env create -f environment.yml` (or `bash scripts/setup_env.sh`).
- [ ] On GPU machine: install CUDA torch build first (see environment.yml note).
- [ ] `bash scripts/setup_env.sh` → editable installs of Trident, HEST, SEAL.
- [x] Copy `.env.example` → `.env`; fill `HF_TOKEN` (path vars still placeholder).
- [~] `python scripts/check_hf_access.py` → **5/6 OK** (only `conchv1_5` pending; see below).
- [ ] HEST downloads bench data automatically on first `benchmark()` run
      (`MahmoodLab/hest-bench`), or pre-fetch it.

## Gated Hugging Face repos needed
- `MahmoodLab/conch`, `MahmoodLab/conchv1_5` — CONCH encoders
- `paige-ai/Virchow2` — Virchow2 encoder
- `MahmoodLab/SEAL` — SEAL checkpoints (bonus)
- `MahmoodLab/hest-bench` (dataset) — eval data
- `MahmoodLab/hest` (dataset) — full HEST-1k (optional)

## HF access check — result (2026-06-13)
Ran `scripts/check_hf_access.py` with a read-scope token (metadata only, no
downloads). **5/6 repos accessible** — everything the main task + SEAL bonus
need is granted; only the optional CONCH v1.5 variant is still pending.

| Repo | Type | Needed for | Status |
|------|------|-----------|--------|
| `MahmoodLab/conch` | model | CONCH v1 (configs' default `conch_v1`) | ✅ OK |
| `paige-ai/Virchow2` | model | Virchow2 | ✅ OK |
| `MahmoodLab/hest-bench` | dataset | benchmark eval data (critical) | ✅ OK |
| `MahmoodLab/hest` | dataset | full HEST-1k (optional) | ✅ OK |
| `MahmoodLab/SEAL` | model | SEAL CONCH/UNIv2 (bonus) | ✅ OK |
| `MahmoodLab/conchv1_5` | model | CONCH v1.5 (optional variant) | ❌ GATED — request access |

Action items:
- Main task is NOT blocked (CONCH v1 + Virchow2 + hest-bench all OK).
- Only if CONCH v1.5 is wanted: request access at
  https://huggingface.co/MahmoodLab/conchv1_5, then re-run the script (→ 6/6).
- Re-run this check on the GPU machine with the same HF account's token.

## Notes
_(your notes here — record which access requests are approved and when)_
