#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="${RUN_ROOT:-outputs/training_log}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
PATIENCE="${PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$RUN_ROOT/logs"

echo "Run root: $RUN_ROOT"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"

python src/training/full_training_process.py \
  --run_root "$RUN_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patience "$PATIENCE" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --naive_losses cross_entropy weighted_cross_entropy ordinal \
  2>&1 | tee -a "$RUN_ROOT/logs/full_training_process.log"

python -m src.analysis.prediction_generator \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  2>&1 | tee -a "$RUN_ROOT/logs/prediction_generator.log"

echo "Vast.ai training and prediction pipeline completed."
