# AGENTS.md

## Cursor Cloud specific instructions

This repo is a **CPU-runnable scaffold** for reproducing HEST-Benchmark Table 1.
There are no servers, databases, or GUIs — it is a batch/CLI ML pipeline. See
`README.md` for the full project description and the canonical command list under
"Usage"; the notes below only cover non-obvious cloud-environment caveats.

### What runs in this environment (no GPU, no HF token, no data)

The update script provisions a Python virtualenv at `.venv/` with the lightweight
CPU stack (`torch` CPU build, `numpy`, `scipy`, `scikit-learn`, `pyyaml`,
`pytest`, `huggingface_hub`). Always invoke tools via `.venv/bin/...`. These all
work with no extra setup:

- `.venv/bin/python -m pytest tests/` — 10 tests (ridge/Pearson math + SEAL
  encoder interface; the SEAL model is mocked, so no weights/network needed).
- `.venv/bin/python scripts/run_benchmark.py --synthetic` — the core
  ridge + Pearson evaluation on synthetic data (expect `pearson_mean ≈ 0.82`).
  This is the best "is it working" smoke test.
- `.venv/bin/python scripts/check_hf_access.py` — metadata-only HF access probe;
  with no `HF_TOKEN` it reports gated repos as FAIL (only the public
  `MahmoodLab/hest-bench` shows OK). That output is expected, not a failure.

### Non-obvious caveats

- **No conda / no Python 3.11 here.** The README/`environment.yml` describe a
  conda env on Python 3.11, but that pin exists only for the GPU-only
  `third_party/` repos (HEST/Trident/SEAL). The CPU scaffold above is
  version-agnostic and runs fine on the system Python 3.12 venv.
- **Lint:** none configured (no ruff/mypy config or deps despite the
  `.ruff_cache`/`.mypy_cache` entries in `.gitignore`). There is nothing to run.
- **GPU/real benchmark is NOT runnable here.** `run_benchmark.py --config ...`,
  `extract_features.py`, and the SEAL `--custom-encoder` path all require: the
  three `third_party/` repos cloned + editable-installed, a CUDA GPU, gated
  Hugging Face access (`HF_TOKEN`), and the tens-to-hundreds-of-GB hest-bench
  dataset. Do not attempt these in the cloud VM; they are documented in
  `README.md` / `PROGRESS.md` for a GPU Linux box.
- `src/encoders/conch_seal_encoder.py` deliberately falls back to a local
  `BasePatchEncoder` shim when `trident` is absent, so it imports with only
  `torch` installed — this is why the encoder tests pass without the GPU stack.
