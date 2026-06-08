#!/usr/bin/env bash
# Full 3^5 per-group hybrid ablation sweep adding the LSTM encoder option.
# Runs all 6 buckets (cmose/daisee/combined x OF-only/+I3D). The runner skips any
# arch_key that already has metrics.json, so the existing 32 (TCN/Transformer-only)
# configs per bucket are preserved and only the 211 LSTM-containing configs train.
#
# CMOSE OpenFace CSVs live under data/CMOSE/secondFeature (resolved as
# secondFeature/<id>.csv) -- not the empty data/CMOSE/features/openface default.
# Matches the path used by evaluate_full_matrix.py, so content is identical to the
# existing CMOSE runs.
#
# Per-sample OpenFace features are memoized to outputs/.feature_cache/openface so the
# first config in each bucket pays the CSV-parse cost and the rest load near-instantly.
# The cache is cleared between datasets to cap disk use (~10 GB peak).
set -u
cd "$(dirname "$0")/.."
PY=.venv-vast-cu131/Scripts/python.exe
COMMON="--num_workers 0 --device auto"
CACHE_DIR=outputs/.feature_cache/openface

run_bucket () {  # dataset model openface_dir labels i3d_dir frames
  local ds="$1" model="$2" of="$3" lbl="$4" i3d="$5" frames="$6"
  echo "=========================================================="
  echo "[$(date '+%F %T')] BUCKET ds=$ds model=$model frames=$frames"
  echo "=========================================================="
  $PY src/training/run_hybrid_ablation.py \
    --dataset "$ds" --model "$model" \
    --feature_dir "$of" --labels_json "$lbl" --i3d_feature_dir "$i3d" \
    --target_frames "$frames" $COMMON
}

run_dataset () {  # dataset openface_dir labels i3d_dir frames
  local ds="$1" of="$2" lbl="$3" i3d="$4" frames="$5"
  run_bucket "$ds" openface_temporal_hybrid     "$of" "$lbl" "$i3d" "$frames"
  run_bucket "$ds" openface_temporal_i3d_hybrid "$of" "$lbl" "$i3d" "$frames"
  echo "[$(date '+%F %T')] clearing feature cache after dataset=$ds"
  rm -rf "$CACHE_DIR"
}

# Order fast -> slow so results flow early.
run_dataset daisee   data/DaiSEE/features/openface   data/DaiSEE/final_data_1.json   data/DaiSEE/features/i3d   300
run_dataset cmose    data/CMOSE/secondFeature        data/CMOSE/final_data_1.json    data/CMOSE/features/i3d    300
run_dataset combined data/combined/features/openface data/combined/final_data_1.json data/combined/features/i3d 150

echo "[$(date '+%F %T')] SWEEP COMPLETE"
