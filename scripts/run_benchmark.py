#!/usr/bin/env python3
"""Run the HEST-Benchmark ridge-regression evaluation.

Two modes:

  1. ``--synthetic`` (CPU, no GPU, no HEST install required)
     Generates random patch features + gene expression and runs the exact
     ridge + Pearson evaluation from ``src.eval.ridge_eval`` end-to-end. This is
     the piece that is verified to work on a Mac and unit-tested in
     ``tests/test_ridge_eval.py``.

  2. real mode (GPU machine, HEST + Trident installed, gated access granted)
     Loads a task config (e.g. configs/brca_conch.yaml), resolves paths from
     configs/paths.yaml, and calls HEST's own ``benchmark()`` which downloads
     the bench data, extracts embeddings via Trident, and runs k-fold ridge
     probing. Optionally swaps in a SEAL custom encoder (bonus task).

Examples:
    # Mac smoke test (the GPU-independent core):
    python scripts/run_benchmark.py --synthetic

    # Real run on the GPU machine:
    python scripts/run_benchmark.py --config configs/brca_conch.yaml

    # Real run with the SEAL bonus encoder:
    python scripts/run_benchmark.py --config configs/brca_conch.yaml \
        --custom-encoder conch_seal
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `src` importable when run from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# --------------------------------------------------------------------------- #
# Mode 1: synthetic smoke test (runs anywhere; the GPU-independent core).
# --------------------------------------------------------------------------- #
def run_synthetic(n_train: int = 400, n_test: int = 120, n_features: int = 384,
                  n_genes: int = 50, dimreduce: str = "PCA", latent_dim: int = 256,
                  seed: int = 1) -> dict:
    """Generate synthetic data with real feature->gene signal and evaluate it.

    Builds ``y = X @ W + noise`` so that a ridge probe should recover a clearly
    positive mean Pearson correlation — a sanity check that the wiring is right.
    """
    import numpy as np

    from src.eval.ridge_eval import run_ridge_eval

    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_features, n_genes))

    def make_split(n):
        X = rng.normal(size=(n, n_features))
        y = X @ W + 0.5 * rng.normal(size=(n, n_genes))
        return X.astype(np.float32), y.astype(np.float32)

    X_train, y_train = make_split(n_train)
    X_test, y_test = make_split(n_test)
    genes = [f"GENE_{i}" for i in range(n_genes)]

    results, _ = run_ridge_eval(
        X_train, X_test, y_train, y_test,
        genes=genes, dimreduce=dimreduce, latent_dim=latent_dim, random_state=seed,
    )
    print("Synthetic HEST-style ridge evaluation")
    print(f"  features={n_features}  genes={n_genes}  "
          f"train={n_train}  test={n_test}  dimreduce={dimreduce}")
    print(f"  pearson_mean = {results['pearson_mean']:.4f}")
    print(f"  pearson_std  = {results['pearson_std']:.4f}")
    top = sorted(results["pearson_corrs"], key=lambda d: d["pearson_corr"],
                 reverse=True)[:3]
    print("  top genes:", ", ".join(f"{d['name']}={d['pearson_corr']:.3f}" for d in top))
    return results


# --------------------------------------------------------------------------- #
# Mode 2: real benchmark via HEST (GPU machine).
# --------------------------------------------------------------------------- #
def build_custom_encoder(name: str, hf_token: str | None):
    """Build a SEAL custom encoder for HEST's ``benchmark(encoder, ...)``."""
    from src.encoders.conch_seal_encoder import build_seal_encoder
    return build_seal_encoder(name, hf_token=hf_token)


def run_real(args: argparse.Namespace) -> None:
    """Load config + paths and dispatch to HEST's benchmark()."""
    from src.utils.config import load_task_config, unresolved_env_vars

    cfg = load_task_config(args.config)

    missing = unresolved_env_vars(cfg)
    if missing:
        print(f"WARNING: unresolved env vars in paths: {sorted(set(missing))}. "
              "Set them in .env (DATA_DIR / RESULTS_DIR).", file=sys.stderr)

    # HEST's benchmark() accepts BenchmarkConfig fields as kwargs (None ignored).
    bench_kwargs = dict(
        datasets=cfg.get("datasets"),
        encoders=[] if args.custom_encoder else cfg.get("encoders"),
        bench_data_root=cfg.get("bench_data_root"),
        embed_dataroot=cfg.get("embed_dataroot"),
        results_dir=cfg.get("results_dir"),
        gene_list=cfg.get("gene_list"),
        method=cfg.get("method"),
        normalize=cfg.get("normalize"),
        dimreduce=cfg.get("dimreduce"),
        latent_dim=cfg.get("latent_dim"),
        batch_size=cfg.get("batch_size"),
        num_workers=cfg.get("num_workers"),
        seed=cfg.get("seed"),
        exp_code=args.exp_code or f"{cfg.get('task', 'task')}_{args.custom_encoder or 'builtin'}",
    )

    custom_encoder = None
    if args.custom_encoder:
        # SEAL bonus: feed our wrapped encoder straight into HEST. The wrapper
        # exposes .eval_transforms and .precision, so enc_transf/precision=None.
        custom_encoder = build_custom_encoder(args.custom_encoder, args.hf_token)

    from hest.bench import benchmark

    dataset_perfs, perf_per_enc = benchmark(
        custom_encoder, None, None, **bench_kwargs
    )
    print("Per-encoder average Pearson:")
    for enc, score in perf_per_enc.items():
        print(f"  {enc}: {score:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="Run the CPU-only synthetic smoke test and exit.")
    p.add_argument("--config", help="Path to a task config (configs/*.yaml).")
    p.add_argument("--custom-encoder", choices=["conch_seal", "univ2_seal"],
                   default=None, help="Use a SEAL custom encoder (bonus task).")
    p.add_argument("--exp-code", default=None, help="Experiment name for outputs.")
    p.add_argument("--hf-token", default=None,
                   help="HF token (else read from HF_TOKEN env var).")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic()
        return 0
    if not args.config:
        print("ERROR: provide --config configs/<task>.yaml (or use --synthetic).",
              file=sys.stderr)
        return 2
    run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
