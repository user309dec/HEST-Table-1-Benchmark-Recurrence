# Day 12 — Comparison with the paper

## Goal
Compare reproduced Pearson scores against HEST Table 1 / the README results
table, and explain any gaps.

## Things that move the numbers
- CONCH **v1** vs **v1.5** (this project defaults to v1 = original Table 1 entry).
- Which CRC task code (COAD vs READ) and whether BRCA = IDC vs LYMPH_IDC.
- PCA on/off and `latent_dim` (256 in `bench_config`).
- Patch size / magnification (256px@20x here vs 112µm@0.5µm/px official).
- Random seed and k-fold split definitions (shipped in the bench data).
- README snapshot date vs the arXiv Table 1 (results were updated over time).

## Comparison table
_(copy the filled results table from day10-11 here and add a Δ column)_

## Notes
_(your analysis here)_
