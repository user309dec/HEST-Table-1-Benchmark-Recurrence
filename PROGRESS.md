# PROGRESS

Two-week plan checklist for reproducing HEST-Benchmark Table 1
(CONCH & Virchow2 on BRCA / CRC / LUAD) + SEAL bonus.

> The detailed task description is in the mentor's `HEST_Benchmark_Task_Summary.md`
> (in the repo root) — see it for the full background, evaluation pseudocode,
> two-week plan, and gotchas.

## Two-week plan
- [ ] **Day 1–2** — Pathology AI review → `notes/day1-2_pathology_ai_review.md`
- [ ] **Day 3–4** — HEST paper notes → `notes/day3-4_hest_paper_notes.md`
- [ ] **Day 5–6** — Environment & data setup → `notes/day5-6_env_data_setup.md`
- [ ] **Day 7** — Trident demo → `notes/day7_trident_demo.md`
- [ ] **Day 8–9** — Feature extraction (CONCH, Virchow2) → `notes/day8-9_feature_extraction.md`
- [ ] **Day 10–11** — Ridge evaluation → `notes/day10-11_ridge_eval.md`
- [ ] **Day 12** — Comparison with paper → `notes/day12_comparison.md`
- [ ] **Day 13–14** — SEAL bonus (CONCH-SEAL / UNIv2-SEAL) → `notes/day13-14_seal_bonus.md`

## Scaffold status (this session, no-GPU Mac)
Done and verified to run on CPU:
- [x] Cloned HEST / Trident / SEAL into `third_party/` (read-only reference).
- [x] Project skeleton (configs / src / scripts / tests / notes).
- [x] `environment.yml` with CUDA notes.
- [x] 6 task configs + central `configs/paths.yaml` (env-driven paths).
- [x] `src/eval/ridge_eval.py` — faithful ridge+Pearson port (**tested on synthetic data**).
- [x] `scripts/run_benchmark.py --synthetic` (**runs on CPU**).
- [x] `src/encoders/conch_seal_encoder.py` — SEAL wrapper on real Trident interface.
- [x] `tests/` — 10 tests passing (`pytest tests/`).
- [x] `scripts/check_hf_access.py` — runs; reports per-repo access (needs token+network).
- [x] `.gitignore` / `.env.example`.

Scaffolded, NOT yet run (needs GPU machine + gated access):
- [ ] `bash scripts/setup_env.sh` — editable installs of the three repos.
- [ ] `scripts/check_hf_access.py` shows all repos OK (run on a networked machine with token).
- [ ] `scripts/extract_features.py` — real Trident extraction.
- [ ] `scripts/run_benchmark.py --config ...` — real HEST benchmark.
- [ ] SEAL weights downloaded; `--custom-encoder conch_seal` end-to-end.

## First thing to do on the GPU machine
See the "Current status / GPU next steps" section of `README.md`.
