#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="${1:-outputs}"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$LOG_DIR"

MODELS=(
  "openface_mlp"
  "temporal_cnn"
  "lstm"
  "transformer"
  "i3d_mlp"
  "openface_tcn_i3d_fusion"
)

COMMON_ARGS=(
  "--epochs" "400"
  "--batch_size" "64"
  "--lr" "1e-4"
)

LOSSES=(
  "cross_entropy"
  "weighted_cross_entropy"
  "ordinal"
)

loss_slug() {
  case "$1" in
    "cross_entropy") echo "ce" ;;
    "weighted_cross_entropy") echo "weighted_ce" ;;
    "ordinal") echo "ordinal" ;;
    *) echo "$1" ;;
  esac
}

echo "Run root: $RUN_ROOT"
echo "Logs: $LOG_DIR"

for model in "${MODELS[@]}"; do
  model_root="$RUN_ROOT/$model"
  mkdir -p "$model_root"

  for loss in "${LOSSES[@]}"; do
    loss_dir_name="$(loss_slug "$loss")"
    output_dir="$model_root/$loss_dir_name"
    log_path="$LOG_DIR/${model}_${loss_dir_name}.log"

    mkdir -p "$output_dir"

    echo "[$(date +"%F %T")] Starting $model with loss=$loss"
    echo "[$(date +"%F %T")] Output: $output_dir" | tee "$log_path"

    python main.py \
      --model "$model" \
      --loss "$loss" \
      --output_dir "$output_dir" \
      "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_path"

    echo "[$(date +"%F %T")] Finished $model with loss=$loss" | tee -a "$log_path"
  done
done

echo "[$(date +"%F %T")] Generating visualizations for $RUN_ROOT"
python scripts/visualize_results.py --outputs_dir "$RUN_ROOT" 2>&1 | tee -a "$LOG_DIR/visualize_results.log"

echo "All comparison runs completed."
