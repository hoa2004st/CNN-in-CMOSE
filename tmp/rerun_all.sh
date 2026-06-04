#!/usr/bin/env bash
# Full rerun driver: train (naive + hybrid) on cmose/daisee/combined, then
# generate the 3x3 prediction matrices (metrics + per-clip CSVs).
# Each dataset is a SEPARATE process so main.py's non-evicting feature cache
# is freed between datasets (avoids OOM on the 32 GB box).
set -u
PY=".venv-vast-cu131/Scripts/python.exe"
LOG_DIR="outputs/training_log/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/rerun_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"; }
run() {
  log "START: $*"
  "$@" >>"$RUN_LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "FAILED (rc=$rc): $*"; else log "OK: $*"; fi
  return 0   # never abort the whole pipeline on a single failure
}

log "===== RERUN START ====="

# ---- Phase 1: naive suite (5 models x 3 losses) per dataset --------------
run "$PY" scripts/compare_naive_models.py --dataset cmose \
    --feature_dir data/CMOSE/features/openface --labels_json data/CMOSE/final_data_1.json \
    --i3d_feature_dir data/CMOSE/features/i3d --target_frames 300 --amp
run "$PY" scripts/compare_naive_models.py --dataset daisee \
    --feature_dir data/DaiSEE/features/openface --labels_json data/DaiSEE/final_data_1.json \
    --i3d_feature_dir data/DaiSEE/features/i3d --target_frames 300 --amp
run "$PY" scripts/compare_naive_models.py --dataset combined \
    --feature_dir data/combined/features/openface --labels_json data/combined/final_data_1.json \
    --i3d_feature_dir data/combined/features/i3d --target_frames 150 --amp

# ---- Phase 2: hybrid ablations (both types, 2^5 sweep) per dataset --------
run "$PY" scripts/run_all_hybrid_ablations.py --datasets cmose   --amp
run "$PY" scripts/run_all_hybrid_ablations.py --datasets daisee  --amp
run "$PY" scripts/run_all_hybrid_ablations.py --datasets combined --amp

# ---- Phase 3: 3x3 predictions (metrics + per-clip CSVs) -------------------
run "$PY" scripts/evaluate_full_matrix.py --device cuda
run "$PY" scripts/evaluate_hybrid_full_matrix.py --device cuda

log "===== RERUN COMPLETE ====="
