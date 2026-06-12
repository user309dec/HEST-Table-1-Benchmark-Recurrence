"""Config loading helpers.

Loads a task config (e.g. ``configs/brca_conch.yaml``) and merges it with the
central ``configs/paths.yaml``, expanding ``${VAR}`` references from the current
environment. Keeping path resolution here means task configs stay machine-agnostic
and only ``configs/paths.yaml`` / ``.env`` change when switching machines.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

_ENV_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` in strings using os.environ."""
    if isinstance(value, str):
        def repl(match: "re.Match[str]") -> str:
            var = match.group(1)
            resolved = os.environ.get(var)
            if resolved is None:
                # Leave the placeholder visible so the user notices the unset var.
                return match.group(0)
            return resolved

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_paths(paths_file: os.PathLike | str | None = None) -> Dict[str, str]:
    """Load configs/paths.yaml with environment variables expanded."""
    paths_file = Path(paths_file) if paths_file else CONFIG_DIR / "paths.yaml"
    raw = load_yaml(paths_file)
    return _expand_env(raw)


def load_task_config(
    config_file: os.PathLike | str,
    paths_file: os.PathLike | str | None = None,
) -> Dict[str, Any]:
    """Load a task config and merge in resolved paths from paths.yaml.

    Returns a flat dict whose keys are a superset of HEST's ``BenchmarkConfig``
    fields (``datasets``, ``encoders``, ``bench_data_root``, ``embed_dataroot``,
    ``results_dir``, ``gene_list``, ``method``, ``normalize``, ``dimreduce``,
    ``latent_dim``, ``batch_size``, ``num_workers``, ``seed`` ...) plus the
    extraction params (``target_magnification``, ``patch_size``, ``overlap``).
    """
    task_cfg = _expand_env(load_yaml(config_file))
    paths = load_paths(paths_file)
    merged: Dict[str, Any] = {}
    merged.update(paths)
    merged.update(task_cfg)  # task config wins on key clashes
    return merged


def unresolved_env_vars(cfg: Dict[str, Any]) -> list[str]:
    """Return any ``${VAR}`` placeholders that were not expanded (for warnings)."""
    found: list[str] = []
    for value in cfg.values():
        if isinstance(value, str):
            found.extend(_ENV_PATTERN.findall(value))
    return found
