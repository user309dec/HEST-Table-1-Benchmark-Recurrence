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

## Comparison
| Task | CONCH | CONCH-SEAL | Δ | Virchow2/UNIv2 | UNIv2-SEAL | Δ |
|------|-------|------------|---|----------------|------------|---|
| BRCA |  |  |  |  |  |  |
| CRC  |  |  |  |  |  |  |
| LUAD |  |  |  |  |  |  |

## Notes
_(your notes here — note SEAL backbones available on HF: conch, univ2)_
