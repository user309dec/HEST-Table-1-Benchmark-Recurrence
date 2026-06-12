# Day 8–9 — Feature extraction

## Goal
Extract patch embeddings for BRCA/CRC/LUAD with CONCH and Virchow2.

Note: when reproducing Table 1 through HEST's `benchmark()`, embedding
extraction happens **inside** HEST from the pre-tiled `hest-bench` data — you do
not need to run Trident extraction separately. Use `scripts/extract_features.py`
only for raw WSIs or the SEAL bonus.

## Encoders
- CONCH → Trident `conch_v1` (or `conch_v15`)
- Virchow2 → Trident `virchow2`

## Sanity checks
- [ ] Embedding `.h5` files written under `$RESULTS_DIR/ST_data_emb/<task>/<enc>/`.
- [ ] Embedding dim sane: CONCH v1 = 512, Virchow2 = 2560 (CLS+mean), conch_v15 = 768.
- [ ] One `.h5` per sample; barcodes present (needed to align with expression).

## Notes
_(record embedding shapes, timings, GPU memory here)_
