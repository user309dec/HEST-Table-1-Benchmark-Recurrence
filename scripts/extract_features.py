#!/usr/bin/env python3
"""Patch feature extraction with Trident (skeleton).

Drives Trident's ``Processor`` to: segment tissue -> patch at 20x/256px ->
extract patch features with a chosen encoder. Mirrors the API shown in
``third_party/trident/tutorials/2-Using-Trident-With-Your-Custom-Patch-Encoder.ipynb``
and ``third_party/trident/run_batch_of_slides.py``.

This GPU-bound step is NOT expected to run on the no-GPU scaffolding machine
(it needs Trident, CUDA, and WSIs). It is structured to import cleanly and to
run as-is once on the GPU box.

NOTE on the HEST benchmark: when you reproduce Table 1 via HEST's own
``benchmark()`` (see scripts/run_benchmark.py), HEST extracts patch embeddings
internally from the pre-tiled HEST-bench data — you do NOT need to run this
script first. This script is for the general Trident pipeline (e.g. extracting
features from your own WSIs, or the SEAL bonus on raw slides).

Example (on the GPU machine):
    python scripts/extract_features.py \
        --wsi-dir $WSI_DIR \
        --job-dir $RESULTS_DIR/trident_out \
        --encoder conch_v1 \
        --mag 20 --patch-size 256

    # SEAL custom encoder (bonus):
    python scripts/extract_features.py --wsi-dir $WSI_DIR --job-dir out \
        --encoder conch_seal
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `src` importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_encoder(encoder_name: str, weights_path: str | None = None,
                  hf_token: str | None = None):
    """Return a Trident-style patch encoder for ``encoder_name``.

    - Built-in Trident encoders (``conch_v1``, ``conch_v15``, ``virchow2``, ...)
      come from ``trident.patch_encoder_models.encoder_factory``.
    - SEAL encoders (``conch_seal``, ``univ2_seal``) come from this repo's
      ``src.encoders.conch_seal_encoder``.
    """
    if encoder_name in ("conch_seal", "univ2_seal"):
        from src.encoders.conch_seal_encoder import build_seal_encoder
        return build_seal_encoder(encoder_name, hf_token=hf_token)

    from trident.patch_encoder_models import encoder_factory
    return encoder_factory(encoder_name, weights_path=weights_path)


def run_extraction(args: argparse.Namespace) -> str:
    """Run segmentation -> patching -> feature extraction. Returns job dir."""
    import torch
    from trident.Processor import Processor
    from trident.segmentation_models import segmentation_model_factory

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.job_dir, exist_ok=True)

    encoder = build_encoder(args.encoder, args.encoder_ckpt, args.hf_token)

    processor = Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
    )

    # 1) tissue vs background segmentation (HEST's segmentation model).
    seg_model = segmentation_model_factory("hest")
    processor.run_segmentation_job(seg_model, device=device)

    # 2) patch coordinate extraction at the requested magnification / size.
    processor.run_patching_job(
        target_magnification=args.mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )

    # 3) patch feature extraction. The coords_dir name encodes the patching
    #    params, e.g. "20x_256px_0px_overlap".
    coords_dir = f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"
    processor.run_patch_feature_extraction_job(
        coords_dir=coords_dir,
        patch_encoder=encoder,
        device=device,
        saveas="h5",
        batch_limit=args.batch_size,
    )
    return args.job_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wsi-dir", default=os.getenv("WSI_DIR"),
                   help="Directory of whole-slide images (default: $WSI_DIR).")
    p.add_argument("--job-dir", required=True, help="Output directory.")
    p.add_argument("--encoder", default="conch_v1",
                   help="Encoder name: a Trident encoder (conch_v1, conch_v15, "
                        "virchow2, ...) or a SEAL encoder (conch_seal, univ2_seal).")
    p.add_argument("--encoder-ckpt", default=None,
                   help="Optional local checkpoint path for the encoder.")
    p.add_argument("--mag", type=int, default=20, help="Target magnification.")
    p.add_argument("--patch-size", type=int, default=256, help="Patch size in px.")
    p.add_argument("--overlap", type=int, default=0, help="Patch overlap in px.")
    p.add_argument("--batch-size", type=int, default=32, help="Feature batch limit.")
    p.add_argument("--hf-token", default=None,
                   help="HF token (else read from HF_TOKEN by the loaders).")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.wsi_dir:
        print("ERROR: --wsi-dir not set (and $WSI_DIR is empty).", file=sys.stderr)
        return 2
    out = run_extraction(args)
    print(f"Done. Features written under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
