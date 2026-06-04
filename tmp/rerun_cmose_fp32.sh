#!/usr/bin/env bash
# cmose failed under AMP: fp16 overflow in FlattenMLP's huge first matmul produced
# NaN eval loss (reproduced: amp=True -> NaN, amp=False -> finite). Re-run cmose in
# fp32 (no --amp), num_workers=0, then regenerate the full 3x3 prediction matrices.
set -u
PY=".venv-vast-cu131/Scripts/python.exe"
LOG_DIR="outputs/training_log/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/rerun_cmose_fp32_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"; }
run() { log "START: $*"; "$@" >>"$RUN_LOG" 2>&1; local rc=$?;
        if [ $rc -ne 0 ]; then log "FAILED (rc=$rc): $*"; else log "OK: $*"; fi; return 0; }

log "===== CMOSE FP32 REDO START ====="

run "$PY" src/training/full_training_process.py --dataset cmose \
    --feature_dir data/CMOSE/features/openface --labels_json data/CMOSE/final_data_1.json \
    --i3d_feature_dir data/CMOSE/features/i3d --target_frames 300 --num_workers 0

run "$PY" src/training/run_all_hybrid_ablations.py --datasets cmose --num_workers 0

rm -f outputs/model_assessment/naive/full_matrix.csv outputs/model_assessment/naive/full_matrix_predictions.csv
rm -f outputs/model_assessment/hybrid/hybrid_matrix.csv outputs/model_assessment/hybrid/hybrid_matrix_predictions.csv
run "$PY" src/evaluation/evaluate_full_matrix.py --device cuda
run "$PY" src/evaluation/evaluate_hybrid_full_matrix.py --device cuda

log "===== CMOSE FP32 REDO COMPLETE ====="
