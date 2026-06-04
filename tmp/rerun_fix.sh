#!/usr/bin/env bash
# Targeted redo: cmose (all) + combined i3d_hybrid failed earlier due to a Windows
# DataLoader shared-memory error (code 1450) with num_workers>0 on large in-RAM
# tensors. Re-run those with num_workers=0, then regenerate the prediction matrices.
set -u
PY=".venv-vast-cu131/Scripts/python.exe"
LOG_DIR="outputs/training_log/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/rerun_fix_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"; }
run() {
  log "START: $*"
  "$@" >>"$RUN_LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "FAILED (rc=$rc): $*"; else log "OK: $*"; fi
  return 0
}

log "===== RERUN FIX START ====="

# ---- cmose naive (num_workers=0) ----
run "$PY" src/training/full_training_process.py --dataset cmose \
    --feature_dir data/CMOSE/features/openface --labels_json data/CMOSE/final_data_1.json \
    --i3d_feature_dir data/CMOSE/features/i3d --target_frames 300 --num_workers 0 --amp

# ---- cmose hybrid, both types (num_workers=0) ----
run "$PY" src/training/run_all_hybrid_ablations.py --datasets cmose --num_workers 0 --amp

# ---- combined i3d_hybrid only; skip-guard keeps the 4 already done (num_workers=0) ----
run "$PY" src/training/run_all_hybrid_ablations.py --datasets combined \
    --model_types openface_temporal_i3d_hybrid --num_workers 0 --amp

# ---- regenerate full 3x3 prediction matrices (clean) ----
rm -f outputs/model_assessment/naive/full_matrix.csv outputs/model_assessment/naive/full_matrix_predictions.csv
rm -f outputs/model_assessment/hybrid/hybrid_matrix.csv outputs/model_assessment/hybrid/hybrid_matrix_predictions.csv
run "$PY" src/evaluation/evaluate_full_matrix.py --device cuda
run "$PY" src/evaluation/evaluate_hybrid_full_matrix.py --device cuda

log "===== RERUN FIX COMPLETE ====="
