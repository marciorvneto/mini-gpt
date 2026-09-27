#!/usr/bin/env bash

set -euo pipefail

echo "=== MiniGPT RunPod Setup ==="

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies..."
uv sync

echo
echo "Checking PyTorch / CUDA..."

uv run python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(props.total_memory / 1024**3, 1),
        "GB",
    )
PY

echo
echo "MiniGPT ready."
