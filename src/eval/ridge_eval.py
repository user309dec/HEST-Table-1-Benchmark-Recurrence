"""Ridge-regression + Pearson evaluation for HEST-Benchmark.

This is a faithful, dependency-light port of the scoring logic in
``third_party/HEST/src/hest/bench/trainer.py`` (``train_test_reg``) plus the
optional PCA dimensionality reduction applied in
``third_party/HEST/src/hest/bench/benchmark.py`` (``predict_single_split``).

It is intentionally decoupled from the heavy HEST/Trident stack so that the
core evaluation can be run and unit-tested on a laptop with no GPU and no
pathology dependencies — only numpy / scipy / scikit-learn are required.

The numbers produced here match HEST's own pipeline because the model
hyper-parameters are replicated exactly:

    alpha = 100 / (n_features * n_genes)
    Ridge(solver='lsqr', alpha=alpha, fit_intercept=False,
          random_state=seed, max_iter=max_iter)

When the GPU machine runs the *real* benchmark through HEST, this module is
not strictly needed — but it is the ground truth we validate against and the
piece that lets us sanity-check the math offline.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import pearsonr


def train_test_reg(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    genes: Optional[Sequence[str]] = None,
    alpha: Optional[float] = None,
    max_iter: int = 1000,
    random_state: int = 0,
):
    """Train Ridge regression and score per-gene Pearson correlation.

    Mirrors ``hest.bench.trainer.train_test_reg`` (``method='ridge'``).

    Args:
        X_train, X_test: patch embeddings, shape ``[n_samples, n_features]``.
        y_train, y_test: gene expression targets, shape ``[n_samples, n_genes]``.
        genes: optional gene names (length ``n_genes``); used only for labelling.
        alpha: Ridge penalty. If ``None``, uses HEST's
            ``100 / (n_features * n_genes)``.
        max_iter: max iterations for the lsqr solver.
        random_state: random seed for the solver.

    Returns:
        (results, dump) where ``results`` is the metrics dict (matching HEST's
        keys: ``pearson_mean``, ``pearson_std``, ``pearson_corrs`` per gene,
        ``l2_errors``, ``r2_scores`` and quantiles) and ``dump`` holds the raw
        predictions/targets.
    """
    from sklearn.linear_model import Ridge

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if alpha is None:
        alpha = 100 / (X_train.shape[1] * y_train.shape[1])

    reg = Ridge(
        solver="lsqr",
        alpha=alpha,
        random_state=random_state,
        fit_intercept=False,
        max_iter=max_iter,
    )
    reg.fit(X_train, y_train)
    preds_all = reg.predict(X_test)

    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    for target in range(y_test.shape[1]):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]
        l2_error = float(np.mean((preds - target_vals) ** 2))
        r2_score = float(
            1
            - np.sum((target_vals - preds) ** 2)
            / np.sum((target_vals - np.mean(target_vals)) ** 2)
        )
        pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corr = float(pearson_corr)

        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        pearson_genes.append(
            {
                "name": genes[target] if genes is not None else f"gene_{target}",
                "pearson_corr": pearson_corr,
            }
        )

    results = {
        "l2_errors": list(errors),
        "r2_scores": list(r2_scores),
        "pearson_corrs": pearson_genes,
        "pearson_mean": float(np.mean(pearson_corrs)),
        "pearson_std": float(np.std(pearson_corrs)),
        "l2_error_q1": float(np.percentile(errors, 25)),
        "l2_error_q2": float(np.median(errors)),
        "l2_error_q3": float(np.percentile(errors, 75)),
        "r2_score_q1": float(np.percentile(r2_scores, 25)),
        "r2_score_q2": float(np.median(r2_scores)),
        "r2_score_q3": float(np.percentile(r2_scores, 75)),
    }
    dump = {"preds_all": preds_all, "targets_all": y_test}
    return results, dump


def run_ridge_eval(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    genes: Optional[Sequence[str]] = None,
    dimreduce: Optional[str] = "PCA",
    latent_dim: int = 256,
    alpha: Optional[float] = None,
    random_state: int = 0,
):
    """Optional PCA reduction, then Ridge + Pearson scoring.

    Mirrors the PCA branch of ``predict_single_split`` in HEST: a
    ``StandardScaler -> PCA(n_components=latent_dim)`` pipeline fit on the train
    split and applied to the test split, before the ridge probe.

    ``latent_dim`` is automatically capped at ``min(n_samples, n_features)`` so
    the function is safe to call on the small synthetic arrays used in tests.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if dimreduce == "PCA":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        n_components = min(latent_dim, X_train.shape[0], X_train.shape[1])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=random_state)),
            ]
        )
        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)

    return train_test_reg(
        X_train,
        X_test,
        y_train,
        y_test,
        genes=genes,
        alpha=alpha,
        random_state=random_state,
    )


def merge_fold_results(fold_results):
    """Aggregate per-fold metrics into k-fold summary (matches HEST).

    See ``hest.bench.benchmark.merge_fold_results``. ``pearson_mean`` is the
    mean of per-fold means; ``pearson_std`` is the std across folds.
    """
    aggr_dict = {}
    for fold in fold_results:
        for item in fold["pearson_corrs"]:
            aggr_dict.setdefault(item["name"], []).append(item["pearson_corr"])

    aggr_results = []
    for name, values in aggr_dict.items():
        aggr_results.append(
            {
                "name": name,
                "pearson_corrs": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )

    mean_per_split = [f["pearson_mean"] for f in fold_results]
    return {
        "pearson_corrs": aggr_results,
        "pearson_mean": float(np.mean(mean_per_split)),
        "pearson_std": float(np.std(mean_per_split)),
        "mean_per_split": mean_per_split,
    }
