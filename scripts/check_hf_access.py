#!/usr/bin/env python3
"""Check Hugging Face access for every gated repo this project needs.

Metadata-only: this never downloads model weights or datasets. It calls the HF
Hub API (``auth_check`` / ``repo_info``) which only touches repo metadata, so it
is safe and fast to run on a laptop. Use it to confirm your access requests went
through *before* moving to the GPU machine.

Usage:
    export HF_TOKEN=hf_...        # or put it in .env
    python scripts/check_hf_access.py

Exit code is non-zero if any required repo is inaccessible.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


# (repo_id, repo_type, why-we-need-it). repo_type: "model" or "dataset".
REQUIRED_REPOS = [
    ("MahmoodLab/conch", "model", "CONCH v1 patch encoder (gated)"),
    ("MahmoodLab/conchv1_5", "model", "CONCH v1.5 patch encoder (gated)"),
    ("paige-ai/Virchow2", "model", "Virchow2 patch encoder (gated)"),
    ("MahmoodLab/SEAL", "model", "SEAL CONCH/UNIv2 checkpoints (gated, bonus)"),
    ("MahmoodLab/hest-bench", "dataset", "HEST-Benchmark eval data (patches/h5ad/splits)"),
    ("MahmoodLab/hest", "dataset", "HEST-1k full dataset (optional, for assembling tasks)"),
]


@dataclass
class Result:
    repo_id: str
    repo_type: str
    note: str
    ok: bool
    detail: str


def _resolve_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_API_KEY")
    )


def check_repo(api, repo_id: str, repo_type: str, note: str, token: str | None) -> Result:
    """Return access status for one repo using metadata-only calls."""
    # Newer huggingface_hub exposes auth_check(); fall back to repo_info().
    try:
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
        )
    except Exception:  # very old hub
        GatedRepoError = RepositoryNotFoundError = Exception  # type: ignore

    try:
        if hasattr(api, "auth_check"):
            api.auth_check(repo_id, repo_type=repo_type, token=token)
        else:
            api.repo_info(repo_id, repo_type=repo_type, token=token)
        return Result(repo_id, repo_type, note, True, "access granted")
    except GatedRepoError:
        return Result(repo_id, repo_type, note, False,
                      "GATED — request access on the repo page, then accept terms")
    except RepositoryNotFoundError:
        return Result(repo_id, repo_type, note, False,
                      "not found OR no access with this token (404 is returned for both)")
    except Exception as exc:  # noqa: BLE001 - surface anything else verbatim
        return Result(repo_id, repo_type, note, False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except Exception:
        print("ERROR: huggingface_hub is not installed. "
              "Run `pip install huggingface_hub` (or create the conda env).")
        return 2

    token = _resolve_token()
    if not token:
        print("WARNING: no HF token found in HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_API_KEY.")
        print("         Gated repos will show as inaccessible. Set one in .env and retry.\n")

    api = HfApi()
    results = [check_repo(api, rid, rtype, note, token)
               for rid, rtype, note in REQUIRED_REPOS]

    width = max(len(r.repo_id) for r in results)
    print(f"{'REPO':<{width}}  {'TYPE':<8}  STATUS   DETAIL")
    print("-" * (width + 40))
    for r in results:
        status = "OK   " if r.ok else "FAIL "
        print(f"{r.repo_id:<{width}}  {r.repo_type:<8}  {status}  {r.detail}")
        print(f"{'':<{width}}  {'':<8}           ({r.note})")

    n_ok = sum(r.ok for r in results)
    print(f"\n{n_ok}/{len(results)} repositories accessible.")
    if n_ok != len(results):
        print("Some repos are not yet accessible. Request access on their HF pages "
              "and make sure your token is registered (`huggingface-cli login`).")
        return 1
    print("All required repositories are accessible. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
