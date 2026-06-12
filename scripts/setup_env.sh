#!/usr/bin/env bash
# Create the conda env and install HEST / Trident / SEAL as editable packages.
#
# Run from the repo root:
#     bash scripts/setup_env.sh
#
# Assumes the three repos are cloned under third_party/ (see README). On the
# GPU machine, install the CUDA build of torch FIRST (see the note in
# environment.yml) so the editable installs below resolve against it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="hest-repro"

# ---- 1. Conda env -----------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    if ! conda env list | grep -q "^${ENV_NAME}\b"; then
        echo "[setup] Creating conda env '${ENV_NAME}' from environment.yml ..."
        conda env create -f environment.yml
    else
        echo "[setup] Conda env '${ENV_NAME}' already exists; updating ..."
        conda env update -f environment.yml --prune
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
else
    echo "[setup] WARNING: conda not found. Continuing in the current Python env."
fi

# ---- 2. third_party present? -----------------------------------------------
for repo in HEST trident SEAL; do
    if [ ! -d "third_party/${repo}" ]; then
        echo "[setup] ERROR: third_party/${repo} missing. Clone the repos first:"
        echo "  mkdir -p third_party && cd third_party"
        echo "  git clone https://github.com/mahmoodlab/HEST.git"
        echo "  git clone https://github.com/mahmoodlab/trident.git"
        echo "  git clone https://github.com/mahmoodlab/SEAL.git"
        exit 1
    fi
done

# ---- 3. GPU torch reminder --------------------------------------------------
echo "[setup] NOTE: on a GPU machine install the CUDA torch build before this,"
echo "        e.g.: pip install torch==2.1.2 torchvision==0.16.2 \\"
echo "                  --index-url https://download.pytorch.org/whl/cu121"

# ---- 4. Editable installs ---------------------------------------------------
# Order matters: Trident first (HEST and SEAL both depend on it), then HEST,
# then SEAL. The [patch-encoders]/[benchmark] extras pull CONCH etc.
echo "[setup] Installing Trident (editable, with patch-encoders extra) ..."
pip install -e "third_party/trident[patch-encoders]"

echo "[setup] Installing HEST (editable, with benchmark extra) ..."
pip install -e "third_party/HEST[benchmark]"

echo "[setup] Installing SEAL (editable) ..."
# SEAL pins exact versions (incl. Linux-only RAPIDS GPU packages). On the GPU
# box prefer `uv sync` inside third_party/SEAL if dependency resolution fights
# pip; --no-deps avoids clobbering the torch/numpy already installed above.
pip install -e "third_party/SEAL" || {
    echo "[setup] Full SEAL install failed (likely RAPIDS/CUDA pins)."
    echo "[setup] Retrying with --no-deps (SEAL code only) ..."
    pip install -e "third_party/SEAL" --no-deps
}

# ---- 5. HF CLI (used by SEAL checkpoint download) ---------------------------
pip install -U "huggingface_hub[cli]"

echo "[setup] Done. Next: copy .env.example to .env, fill HF_TOKEN/DATA_DIR/RESULTS_DIR,"
echo "        then run: python scripts/check_hf_access.py"
