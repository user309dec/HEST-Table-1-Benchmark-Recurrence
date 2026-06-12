"""Tests for the ridge + Pearson evaluation (the GPU-independent core).

These run on a laptop with only numpy / scipy / scikit-learn installed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.ridge_eval import (  # noqa: E402
    merge_fold_results,
    run_ridge_eval,
    train_test_reg,
)


def _make_data(n_train=300, n_test=100, n_features=64, n_genes=50,
               noise=0.5, seed=0):
    """y = X @ W + noise so a ridge probe should recover positive correlation."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_features, n_genes))

    def split(n):
        X = rng.normal(size=(n, n_features))
        y = X @ W + noise * rng.normal(size=(n, n_genes))
        return X.astype(np.float32), y.astype(np.float32)

    Xtr, ytr = split(n_train)
    Xte, yte = split(n_test)
    return Xtr, Xte, ytr, yte


def test_results_structure_and_shapes():
    Xtr, Xte, ytr, yte = _make_data()
    genes = [f"g{i}" for i in range(ytr.shape[1])]
    results, dump = train_test_reg(Xtr, Xte, ytr, yte, genes=genes)

    # Metric keys match HEST's trainer.train_test_reg output.
    for key in ("pearson_mean", "pearson_std", "pearson_corrs",
                "l2_errors", "r2_scores"):
        assert key in results
    assert len(results["pearson_corrs"]) == ytr.shape[1]
    assert len(results["l2_errors"]) == ytr.shape[1]
    assert results["pearson_corrs"][0]["name"] == "g0"

    # Predictions line up with the test split.
    assert dump["preds_all"].shape == yte.shape
    assert -1.0 <= results["pearson_mean"] <= 1.0


def test_signal_is_recovered():
    """With real X->y signal, mean Pearson should be clearly positive."""
    Xtr, Xte, ytr, yte = _make_data(noise=0.3)
    results, _ = train_test_reg(Xtr, Xte, ytr, yte)
    assert results["pearson_mean"] > 0.5


def test_default_alpha_matches_hest_formula():
    """alpha defaults to 100 / (n_features * n_genes), per HEST."""
    Xtr, Xte, ytr, yte = _make_data(n_features=64, n_genes=50)
    expected_alpha = 100 / (Xtr.shape[1] * ytr.shape[1])

    captured = {}
    import sklearn.linear_model as lm
    real_ridge = lm.Ridge

    def spy(*args, **kwargs):
        captured["alpha"] = kwargs.get("alpha")
        captured["solver"] = kwargs.get("solver")
        captured["fit_intercept"] = kwargs.get("fit_intercept")
        return real_ridge(*args, **kwargs)

    lm.Ridge = spy
    try:
        train_test_reg(Xtr, Xte, ytr, yte)
    finally:
        lm.Ridge = real_ridge

    assert captured["alpha"] == pytest.approx(expected_alpha)
    assert captured["solver"] == "lsqr"
    assert captured["fit_intercept"] is False


def test_pca_reduction_runs_and_caps_components():
    """PCA branch should run and cap n_components at min(n_samples, n_features)."""
    Xtr, Xte, ytr, yte = _make_data(n_train=80, n_features=64)
    # latent_dim larger than n_samples -> must be capped, not error.
    results, _ = run_ridge_eval(Xtr, Xte, ytr, yte, dimreduce="PCA",
                                latent_dim=256)
    assert results["pearson_mean"] > 0.3


def test_no_dimreduce_matches_direct_call():
    Xtr, Xte, ytr, yte = _make_data()
    r1, _ = run_ridge_eval(Xtr, Xte, ytr, yte, dimreduce=None)
    r2, _ = train_test_reg(Xtr, Xte, ytr, yte)
    assert r1["pearson_mean"] == pytest.approx(r2["pearson_mean"])


def test_merge_fold_results():
    """k-fold aggregation matches HEST: mean of per-fold means."""
    fold_a = {"pearson_mean": 0.4,
              "pearson_corrs": [{"name": "g0", "pearson_corr": 0.3},
                                {"name": "g1", "pearson_corr": 0.5}]}
    fold_b = {"pearson_mean": 0.6,
              "pearson_corrs": [{"name": "g0", "pearson_corr": 0.7},
                                {"name": "g1", "pearson_corr": 0.5}]}
    merged = merge_fold_results([fold_a, fold_b])
    assert merged["pearson_mean"] == pytest.approx(0.5)
    assert merged["mean_per_split"] == [0.4, 0.6]
    g0 = next(g for g in merged["pearson_corrs"] if g["name"] == "g0")
    assert g0["mean"] == pytest.approx(0.5)
