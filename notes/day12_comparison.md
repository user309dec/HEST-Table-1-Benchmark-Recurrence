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

## Comparison table (reproduced on RTX 4080, 2026-06-17)
4-fold mean Pearson, official HEST-bench splits, PCA(256) + Ridge, top-50 HVG.

| Task (HEST code) | conch_v1 (ours) | virchow2 (ours) | HEST ref (CONCH) | Δ vs ref |
|------------------|-----------------|-----------------|------------------|----------|
| BRCA / IDC       | **0.5363**      | **0.5971**      | 0.5363 (README)  | **0.000** (exact) |
| CRC / COAD       | 0.2487          | 0.258           | —                | — |
| CRC / READ       | 0.1601          | 0.2072          | —                | — |
| CRC (COAD/READ avg) | **0.2044**   | **0.2326**      | ~0.18–0.25 (mentor §八, rough) | within range |
| LUAD / LUNG      | **0.5323**      | **0.5685**      | —                | — |

Per-fold BRCA/IDC: CONCH 0.4107/0.5994/0.51/0.6253; Virchow2 0.4811/0.6422/0.5615/0.7036.

## Notes
- **BRCA/IDC CONCH matches the HEST README/leaderboard value (0.5363) exactly** —
  strong evidence the pipeline is faithful (same encoder, PCA, ridge alpha,
  splits).
- **Virchow2 > CONCH on every task** (BRCA +0.061, CRC +0.028, LUAD +0.036),
  consistent with the paper's ordering (Virchow2 ViT-H is a stronger encoder).
- **Open item for the mentor (flagged in README caveats):** mentor §八 lists
  CONCH BRCA ~0.20–0.28 as a "rough magnitude", but the actual HEST value is
  0.5363 (which we reproduced). §八 itself says "具体数值以原论文 Table 1 为准";
  so §八 was just a low estimate, not the target. CRC, by contrast, genuinely is
  in the low-0.2 range and matches §八.
- READ (0.16) is the hardest sub-task (smaller cohort, lower signal) and drags
  the CRC average below COAD (0.25).
- All runs used `conch_v1` (original Table-1 CONCH) and 256px@20x; the official
  data is tiled at 112µm@0.5µm/px — a minor protocol difference that does not
  prevent matching the headline BRCA number.
