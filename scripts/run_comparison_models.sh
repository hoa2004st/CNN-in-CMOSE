#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ROOT="${1:-outputs/comparison_runs/$TIMESTAMP}"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$LOG_DIR"

MODELS=(
  "temporal_cnn"
  "rectangular_cnn"
  "lstm"
  "transformer"
)

COMMON_ARGS=(
  "--epochs" "1600"
  "--batch_size" "8"
  "--lr" "1e-4"
)

echo "Run root: $RUN_ROOT"
echo "Logs: $LOG_DIR"

for model in "${MODELS[@]}"; do
  output_dir="$RUN_ROOT/$model"
  log_path="$LOG_DIR/$model.log"

  mkdir -p "$output_dir"

  echo "[$(date +"%F %T")] Starting $model"
  echo "[$(date +"%F %T")] Output: $output_dir" | tee "$log_path"

  python main.py \
    --model "$model" \
    --output_dir "$output_dir" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_path"

  echo "[$(date +"%F %T")] Finished $model" | tee -a "$log_path"
done

echo "All comparison runs completed."
