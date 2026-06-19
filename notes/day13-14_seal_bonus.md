# Day 13–14 — SEAL bonus

## Goal
Integrate SEAL's CONCH-SEAL / UNIv2-SEAL as custom encoders into the
Trident/HEST feature-extraction + benchmark pipeline, and compare against the
vanilla CONCH / Virchow2 baselines.

## How it's wired (this repo)
- `src/encoders/conch_seal_encoder.py`:
  - `CONCHSEALEncoder` / `UNIv2SEALEncoder` wrap SEAL's `seal_factory` in
    Trident's `BasePatchEncoder` interface (`_build` → `(model, transforms,
    precision)`; `forward` → `[B, D]`).
- Run through HEST:
  ```bash
  python scripts/run_benchmark.py --config configs/brca_conch.yaml \
      --custom-encoder conch_seal
  ```
- Or through Trident directly:
  ```bash
  python scripts/extract_features.py --wsi-dir $WSI_DIR --job-dir out \
      --encoder conch_seal
  ```

## Weights (gated — do this first)
1. Request access: https://huggingface.co/MahmoodLab/SEAL
2. `export HF_TOKEN=hf_...`
3. `seal_factory(backbone="conch", source="auto")` downloads
   `seal_conch_vision.pth` + `seal_conch_omics.pth` into `weights/conch_SEAL/`.
   (See the `TODO(weights)` in `SEALPatchEncoder._build`.)

## Comparison (reproduced on RTX 4080, 2026-06-17/18)
| Task | CONCH (conch_v1) | CONCH-SEAL (conch_seal) | Δ (SEAL−CONCH) | Virchow2 |
|------|------------------|-------------------------|----------------|----------|
| BRCA / IDC | 0.5363 | **0.5484** | **+0.0121** | 0.5971 |
| CRC (COAD/READ avg) | 0.2044 | **0.2245** | **+0.0201** | 0.2326 |
| LUAD / LUNG | 0.5323 | **0.5349** | **+0.0026** | 0.5685 |

CONCH-SEAL per-fold (BRCA/IDC): 0.4391 / 0.5942 / 0.5329 / 0.6275.
**SEAL improves CONCH on all three tasks** (Δ = +0.012 / +0.020 / +0.003), with
the largest gain on CRC. It still trails Virchow2 (a much larger ViT-H backbone).

## Notes
- **SEAL's ST-molecular fine-tuning of CONCH helps**: CONCH-SEAL 0.5484 > CONCH
  0.5363 on BRCA/IDC (+0.012), the bonus's expected direction — injecting spatial
  transcriptomic signal into the backbone yields features that better predict
  expression.
- CONCH-SEAL is a **LoRA-adapted CONCH ViT-B/16** (only ~98k trainable params in
  the adapter, on the last transformer block). It still trails Virchow2 (0.5971),
  which is a far larger ViT-H model — so backbone scale still dominates here.
- `conch_seal` was run on **all three tasks** (BRCA/CRC/LUAD). The bonus asks for
  CONCH-SEAL **or** UNIv2-SEAL — done with CONCH-SEAL. `univ2_seal` weights exist
  on HF (`seal_univ2_{vision,omics}.pth`) but need UNI2 backbone access (easy
  follow-up: `--custom-encoder univ2_seal`).

## Environment gotchas (so this is reproducible)
SEAL pins transformers 4.48 + timm 1.0.9, which conflict with the main stack
(transformers 4.40.2 + timm 0.9.16 for torch 2.1.2). Resolution used:
- Did NOT install SEAL's full deps. Added only `peft==0.11.1 accelerate==0.30.1`
  (`--no-deps`) + `setuptools<81` (restores `pkg_resources`). Kept
  `huggingface_hub==0.36.2` (<1.0, required by transformers 4.40.2; ships `hf` CLI).
- Ran the benchmark with **CWD = `third_party/SEAL`** and **`.venv/bin` on PATH**
  so SEAL's relative `conf/config.yaml` + `weights/` and the `hf` download CLI
  resolve. Checkpoints auto-download to `weights/conch_SEAL/`.
- Helper script: `logs/_run_seal3.sh`.
