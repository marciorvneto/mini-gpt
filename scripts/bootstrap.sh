#!/usr/bin/env bash

set -euo pipefail

REPO_URL="https://github.com/marciorvneto/mini-gpt.git"
REPO_DIR="/root/mini-gpt"

export HOME="/root"
export XDG_CACHE_HOME="/root/.cache"
export UV_CACHE_DIR="/root/.cache/uv"
export MINIGPT_DATA_ROOT="/workspace/data"
export MINIGPT_RUN_ROOT="/workspace/runs"

mkdir -p "$UV_CACHE_DIR"
mkdir -p "$MINIGPT_DATA_ROOT"
mkdir -p "$MINIGPT_RUN_ROOT"

# ------------------------------------------------------------
# uv
# ------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi

# ------------------------------------------------------------
# Source
# ------------------------------------------------------------

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

git fetch origin
git checkout main
git pull --ff-only

# ------------------------------------------------------------
# Exact Python environment
# ------------------------------------------------------------

uv python install 3.13.8

uv sync \
    --frozen \
    --python 3.13.8

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

uv run python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(p.total_memory / 1024**3, 1), "GB")
PY

echo
echo "MiniGPT ready."
