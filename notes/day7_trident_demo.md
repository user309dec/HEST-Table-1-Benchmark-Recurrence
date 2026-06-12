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

## Notes
_(your notes here)_
