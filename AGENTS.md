# AGENTS.md

## Cursor Cloud specific instructions

This repo reproduces a subset of the HEST-Benchmark (gene-expression prediction
from H&E patches). It splits cleanly into a **CPU-runnable core** and a
**GPU-only full benchmark**. See `README.md` for the full picture; the notes
below only cover what is non-obvious for working in this environment.

### Scope that runs here (no GPU)
The cloud VM has **no GPU**, no gated Hugging Face access, and not the
tens-to-hundreds of GB of HEST-bench data. So only the GPU-independent pieces
run here — and these are exactly what the project documents as CPU-verified:
- `pytest tests/` — 10 tests (ridge/Pearson math + SEAL encoder interface).
- `python scripts/run_benchmark.py --synthetic` — full ridge+Pearson eval on
  synthetic data (the core "does the wiring work" smoke test).
- `python scripts/check_hf_access.py` — metadata-only HF access report.
- Config loading via `src/utils/config.py` (env-var path expansion).

**Out of scope here (GPU machine only, do not attempt to run/install):**
`scripts/extract_features.py`, real-mode `scripts/run_benchmark.py --config ...`,
the SEAL `--custom-encoder` path, and the editable installs of
`third_party/{HEST,trident,SEAL}`. These need CUDA torch, RAPIDS, gated weights,
and the bench dataset. `scripts/setup_env.sh` + `environment.yml` are for that
GPU box, **not** for this VM.

### How dependencies are set up here
There is no conda on this VM and no `requirements.txt`. The CPU subset of
`environment.yml` is installed into a local virtualenv at `.venv/` (git-ignored)
by the startup update script. The system package `python3-venv` is required to
create the venv and is installed at the OS level (persisted in the VM snapshot),
so the update script itself only creates the venv and `pip install`s.

- Always run Python through the venv: `.venv/bin/python ...` (or activate with
  `source .venv/bin/activate`).
- `huggingface_hub` installs as a 1.x release here; that is fine for
  `check_hf_access.py` (it only uses `HfApi.auth_check`). The README's note about
  pinning hub `<1.0` applies only to the GPU stack (transformers 4.40.2), which
  is not installed here.

### Non-obvious gotchas
- `check_hf_access.py` exits non-zero (1) when any gated repo is inaccessible.
  Without an `HF_TOKEN` secret this is expected and **not** an environment
  failure — the script reaching the HF API and reporting per-repo status is the
  success condition. The public `MahmoodLab/hest-bench` dataset shows `OK`; the
  gated repos show `FAIL`/`GATED`. Provide an `HF_TOKEN` env var (read scope,
  with access granted on each repo page) to turn the gated repos green.
- Scripts add the repo root to `sys.path` themselves, so run them from the repo
  root (`/workspace`); there is no installed package for `src/`.
