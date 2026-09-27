#!/usr/bin/env bash

set -euo pipefail

export MINIGPT_DATA_ROOT="${MINIGPT_DATA_ROOT:-data}"
export MINIGPT_RUN_ROOT="${MINIGPT_RUN_ROOT:-.}"

uv run train_v2.py \
  --micro-batch-size 2 \
  --grad-accum-steps 8 \
  "$@"
