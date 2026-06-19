# Day 7 — Trident demo

## Goal
Run Trident's `Processor` pipeline once on a sample WSI to understand
segmentation → patching → feature extraction, and confirm the encoder interface.

## Pipeline (from `scripts/extract_features.py` and the Trident tutorial)
1. `Processor(job_dir=..., wsi_source=...)`
2. `run_segmentation_job(segmentation_model_factory('hest'), device)`
3. `run_patching_job(target_magnification=20, patch_size=256, overlap=0)`
   → produces a `20x_256px_0px_overlap` coords dir.
4. `run_patch_feature_extraction_job(coords_dir=..., patch_encoder=encoder, saveas='h5')`

## Encoder interface (Trident `BasePatchEncoder`)
- `_build()` returns `(model, eval_transforms, precision)`.
- `forward(x)` defaults to `self.model(x)`.
- Custom models: `CustomInferenceEncoder(enc_name, model, transforms, precision)`.

## Run
```bash
python scripts/extract_features.py --wsi-dir $WSI_DIR --job-dir out --encoder conch_v1
```

## Notes — what we actually ran (2026-06-18)
- For **Table 1 reproduction we did NOT need the segmentation + patching steps**:
  `MahmoodLab/hest-bench` ships **pre-tiled patches** (`<task>/patches/*.h5`)
  already registered to the ST spots. So steps 1–3 above (segmentation →
  patching) are skipped, and only **step 4 (patch feature extraction)** runs —
  and it runs *inside* HEST's `benchmark()`, which calls Trident's
  `encoder_factory(<name>)` to load `conch_v1` / `virchow2` and embed each
  patch on the GPU. Embeddings are cached under `RESULTS_DIR/ST_data_emb/`.
- This is why "Day 7 Trident demo on a raw WSI" is optional for this project:
  the standalone `extract_features.py` (seg→patch→extract) is only needed if you
  bring **your own WSIs**, not for the benchmark.
- The **encoder interface is the part that matters for the bonus**: our SEAL
  wrapper (`src/encoders/conch_seal_encoder.py`) implements exactly the
  `BasePatchEncoder` contract above (`_build → (model, transforms, precision)`,
  `forward → [B, D]`), so HEST/Trident drive it identically to a native encoder.
  Verified by `tests/test_encoder_interface.py` (4 tests) and by the real
  `--custom-encoder conch_seal` run.
- GPU-cost observation: CONCH (ViT-B/16) extraction over a task is ~minutes at
  batch 128; Virchow2 (ViT-H/14) needs batch_size 8 to fit 12 GB but still runs
  without OOM.

## Sources
- Trident: github.com/mahmoodlab/trident (Processor + patch_encoder_models).
- Encoder interface read from `trident/patch_encoder_models/load.py`.
