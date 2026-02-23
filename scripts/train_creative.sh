#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cargo run --bin train --release -- \
  --domain creative \
  --data data/creative.txt \
  --out weights \
  --steps "${STEPS:-30000}" \
  --vocab-size "${VOCAB_SIZE:-400}" \
  --lr "${LR:-5e-4}" \
  --eval-every "${EVAL_EVERY:-500}"
