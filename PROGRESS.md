# PROGRESS

Two-week plan checklist for reproducing HEST-Benchmark Table 1
(CONCH & Virchow2 on BRCA / CRC / LUAD) + SEAL bonus.

> The detailed task description is in the mentor's `HEST_Benchmark_Task_Summary.md`
> (in the repo root) — see it for the full background, evaluation pseudocode,
> two-week plan, and gotchas.

## Two-week plan
- [x] **Day 1–2** — Pathology AI review → `notes/day1-2_pathology_ai_review.md` (formal, web-verified)
- [x] **Day 3–4** — HEST paper notes → `notes/day3-4_hest_paper_notes.md` (formal, web-verified)
- [x] **Day 5–6** — Environment & data setup → env built on RTX 4080, hest-bench IDC/COAD/READ/LUNG downloaded
- [x] **Day 7** — Trident demo → `notes/day7_trident_demo.md` (documented; HEST data pre-tiled so only feature-extraction stage runs)
- [x] **Day 8–9** — Feature extraction (CONCH, Virchow2) → embeddings cached for all task×encoder combos
- [x] **Day 10–11** — Ridge evaluation → all 6 main runs scored
- [x] **Day 12** — Comparison with paper → `notes/day12_comparison.md` filled with real numbers
- [x] **Day 13–14** — SEAL bonus (CONCH-SEAL) → `notes/day13-14_seal_bonus.md` filled; CONCH-SEAL = 0.5484

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
- [x] Env built on RTX 4080 (12GB) via `uv` venv (py3.11), torch 2.1.2+cu121, numpy<2, transformers 4.40.2.
- [x] Editable installs of trident[patch-encoders] + HEST (SEAL deferred — bonus).
- [x] `scripts/check_hf_access.py` — **6/6 repos OK**.
- [x] `scripts/run_benchmark.py --config configs/brca_conch.yaml` — **real HEST benchmark ran end-to-end**.
- [x] `brca_virchow2.yaml` (batch_size 8 for 12GB) — ran end-to-end, no OOM.
- [x] CRC / LUAD tasks — all ran (luad_conch needed a rerun after an hf_hub version clash).
- [x] SEAL `--custom-encoder conch_seal` end-to-end — **ran on all 3 tasks** (see notes).

## Results (real benchmark, RTX 4080) — main task complete
| Task | conch_v1 | virchow2 |
|------|----------|----------|
| BRCA / IDC | **0.5363** | **0.5971** |
| CRC (COAD/READ avg) | **0.2044** (COAD .249 / READ .160) | **0.2326** (COAD .258 / READ .207) |
| LUAD / LUNG | **0.5323** | **0.5685** |

Virchow2 > CONCH on every task (as in the paper). BRCA/IDC conch_v1 = 0.5363 matches the HEST README value exactly.

### SEAL bonus — DONE (CONCH-SEAL on all 3 tasks)
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

## First thing to do on the GPU machine
See the "Current status / GPU next steps" section of `README.md`.
