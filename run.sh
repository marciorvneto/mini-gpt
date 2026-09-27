#!/usr/bin/env bash

set -euo pipefail

export MINIGPT_DATA_ROOT="${MINIGPT_DATA_ROOT:-/workspace/data}"
export MINIGPT_RUN_ROOT="${MINIGPT_RUN_ROOT:-/workspace/runs}"

uv run train_v2.py \
  --micro-batch-size 16 \
  --grad-accum-steps 1 \
  "$@"
