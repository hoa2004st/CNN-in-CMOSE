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

python scripts/compare_naive_models.py \
  --run_root "$RUN_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --patience "$PATIENCE" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --naive_losses cross_entropy weighted_cross_entropy ordinal \
  2>&1 | tee -a "$RUN_ROOT/logs/compare_naive_models.log"

python -m src.feature_analysis.run_domain_shift_analysis \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  2>&1 | tee -a "$RUN_ROOT/logs/run_domain_shift_analysis.log"

python scripts/visualize_model_assessment.py \
  2>&1 | tee -a "$RUN_ROOT/logs/visualize_model_assessment.log"

python scripts/visualize_dataset_analysis.py \
  2>&1 | tee -a "$RUN_ROOT/logs/visualize_dataset_analysis.log"

python scripts/feature_space_dataset_comparison.py \
  2>&1 | tee -a "$RUN_ROOT/logs/feature_space_dataset_comparison.log"

python scripts/visualize_feature_space_domains.py \
  2>&1 | tee -a "$RUN_ROOT/logs/visualize_feature_space_domains.log"

echo "Vast.ai training, evaluation, and visualization pipeline completed."
