#!/usr/bin/env bash
# =============================================================================
# DAiSEE feature-extraction pipeline for a vast.ai Ubuntu GPU instance.
#
# End to end it: clones this repo, installs deps, downloads DAiSEE from Kaggle,
# extracts OpenFace (709-dim) + I3D (1024-dim) features in the same format the
# CMOSE pipeline consumes, builds the engagement labels, then publishes:
#   * small label files  -> this GitHub repo  (committed)
#   * large feature files -> a private Kaggle dataset (too big for git)
#
# Run the whole thing, or a subset of stages, via the STAGES variable.
# Designed to be re-run: each stage is guarded and skips finished work.
#
# Stages: repo env kaggle labels openface i3d push
#
# Usage (on the remote box):
#   export GITHUB_TOKEN=ghp_xxx GIT_USER_EMAIL=you@x GIT_USER_NAME="You"
#   export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
#   bash scripts/daisee_extract_vast.sh
#   STAGES="openface i3d" bash scripts/daisee_extract_vast.sh   # rerun a subset
# =============================================================================
set -euo pipefail

# ----------------------------- configuration ---------------------------------
REPO_URL="${REPO_URL:-https://github.com/hoa2004st/CNN-in-CMOSE}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/CNN-in-CMOSE}"

KAGGLE_DATASET="${KAGGLE_DATASET:-olgaparfenova/daisee}"     # source dataset
DAISEE_ROOT="${DAISEE_ROOT:-data/DaiSEE}"
RAW_DIR="${RAW_DIR:-$DAISEE_ROOT/raw}"                       # extracted Kaggle zip
OPENFACE_DIR="$DAISEE_ROOT/features/openface"
I3D_DIR="$DAISEE_ROOT/features/i3d"
CLIPS_TXT="$DAISEE_ROOT/clips.txt"

JOBS="${JOBS:-$(nproc)}"                                     # OpenFace parallelism
OPENFACE_IMAGE="${OPENFACE_IMAGE:-algebr/openface:latest}"
I3D_CKPT_URL="https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth"

# Where to publish the extracted features (private Kaggle dataset).
# Defaults to "<your-kaggle-user>/daisee-cmose-features".
FEATURES_KAGGLE_SLUG="${FEATURES_KAGGLE_SLUG:-${KAGGLE_USERNAME:-unknown}/daisee-cmose-features}"

STAGES="${STAGES:-repo env kaggle labels openface i3d push}"

# ------------------------------- helpers --------------------------------------
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }
want() { [[ " $STAGES " == *" $1 "* ]]; }

VENV_PY=""   # set by the env stage

# ------------------------------- stage: repo ----------------------------------
stage_repo() {
  log "Cloning / updating repo at $WORKDIR"
  if [[ -d "$WORKDIR/.git" ]]; then
    git -C "$WORKDIR" fetch origin "$REPO_BRANCH"
    git -C "$WORKDIR" checkout "$REPO_BRANCH"
    git -C "$WORKDIR" pull --ff-only origin "$REPO_BRANCH" || warn "pull skipped (local changes?)"
  else
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
  fi
}

# ------------------------------- stage: env -----------------------------------
stage_env() {
  log "Creating base Python env (.venv) + CUDA torch + project deps"
  cd "$WORKDIR"
  command -v python3 >/dev/null || die "python3 not found"
  [[ -d .venv ]] || python3 -m venv .venv
  VENV_PY="$WORKDIR/.venv/bin/python"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
  # CUDA 12.1 torch wheels (matches scripts/setup_vast_cuda_env.ps1).
  "$VENV_PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  "$VENV_PY" -m pip install -r requirements.txt
  "$VENV_PY" -m pip install kaggle
  "$VENV_PY" -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
}

resolve_venv_py() {
  VENV_PY="$WORKDIR/.venv/bin/python"
  [[ -x "$VENV_PY" ]] || die "base venv missing; run the 'env' stage first"
}

# ------------------------------ stage: kaggle ---------------------------------
# DAiSEE source download. Needs Kaggle API creds. If you don't have a token:
#   1. Sign in at https://www.kaggle.com -> Account -> "Create New API Token".
#   2. That downloads kaggle.json. Either place it at ~/.kaggle/kaggle.json
#      (chmod 600), or export KAGGLE_USERNAME / KAGGLE_KEY before running.
configure_kaggle() {
  mkdir -p "$HOME/.kaggle"
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$HOME/.kaggle/kaggle.json"
  fi
  [[ -f "$HOME/.kaggle/kaggle.json" ]] || die "No Kaggle credentials. Set KAGGLE_USERNAME/KAGGLE_KEY or drop kaggle.json in ~/.kaggle/"
  chmod 600 "$HOME/.kaggle/kaggle.json"
}

stage_kaggle() {
  resolve_venv_py
  cd "$WORKDIR"
  configure_kaggle
  log "Downloading DAiSEE ($KAGGLE_DATASET) into $RAW_DIR (this is large: budget 50+ GB free disk)"
  mkdir -p "$RAW_DIR"
  if find "$RAW_DIR" -iname '*train*label*.csv' | grep -q .; then
    warn "Label CSVs already present under $RAW_DIR; skipping download/unzip"
    return
  fi
  "$VENV_PY" -m kaggle datasets download -d "$KAGGLE_DATASET" -p "$RAW_DIR"
  log "Unzipping archives"
  for zip in "$RAW_DIR"/*.zip; do
    [[ -e "$zip" ]] || continue
    "$VENV_PY" - "$zip" "$RAW_DIR" <<'PY'
import sys, zipfile
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    z.extractall(dest)
print("extracted", zip_path)
PY
    rm -f "$zip"
  done
}

build_clips_txt() {
  log "Indexing DAiSEE video clips"
  mkdir -p "$DAISEE_ROOT"
  find "$RAW_DIR" -type f \( -iname '*.avi' -o -iname '*.mp4' -o -iname '*.mov' \) | sort > "$CLIPS_TXT"
  local count
  count="$(wc -l < "$CLIPS_TXT")"
  [[ "$count" -gt 0 ]] || die "No video clips found under $RAW_DIR"
  log "Found $count clips -> $CLIPS_TXT"
}

# ------------------------------ stage: labels ---------------------------------
stage_labels() {
  resolve_venv_py
  cd "$WORKDIR"
  log "Building CMOSE-compatible engagement labels"
  "$VENV_PY" scripts/build_daisee_labels.py \
    --dataset-root "$RAW_DIR" \
    --out-json "$DAISEE_ROOT/final_data_1.json" \
    --out-csv  "$DAISEE_ROOT/labels.csv"
}

# ----------------------------- stage: openface --------------------------------
# OpenFace FeatureExtraction with no feature-restricting flags emits all 709
# features (714 cols incl. metadata) -> matches CMOSE exactly. Output file is
# named after the input stem; we append "_person0" so the CMOSE loader's
# `sample_id.rsplit("_person", 1)` works unchanged.
stage_openface() {
  cd "$WORKDIR"
  [[ -f "$CLIPS_TXT" ]] || build_clips_txt
  mkdir -p "$OPENFACE_DIR"

  local raw_out="$DAISEE_ROOT/_openface_raw"
  mkdir -p "$raw_out"

  # Resolve a FeatureExtraction invocation: native binary, else docker image.
  local of_native=""
  if command -v FeatureExtraction >/dev/null 2>&1; then
    of_native="$(command -v FeatureExtraction)"
  elif [[ -x /home/openface-build/build/bin/FeatureExtraction ]]; then
    of_native="/home/openface-build/build/bin/FeatureExtraction"
  fi

  if [[ -n "$of_native" ]]; then
    log "OpenFace via native binary: $of_native (JOBS=$JOBS)"
    OF_BIN="$of_native" RAW_OUT="$raw_out" OPENFACE_DIR="$OPENFACE_DIR" \
      bash -c '
        export OF_BIN RAW_OUT OPENFACE_DIR
        run_one() {
          local clip="$1"; local stem; stem="$(basename "${clip%.*}")"
          [[ -f "$OPENFACE_DIR/${stem}_person0.csv" ]] && return 0
          "$OF_BIN" -f "$clip" -out_dir "$RAW_OUT" -q >/dev/null 2>&1 || { echo "FAIL $clip" >&2; return 0; }
          [[ -f "$RAW_OUT/${stem}.csv" ]] && mv -f "$RAW_OUT/${stem}.csv" "$OPENFACE_DIR/${stem}_person0.csv"
          rm -f "$RAW_OUT/${stem}_of_details.txt"
        }
        export -f run_one
        cat "'"$CLIPS_TXT"'" | xargs -P "'"$JOBS"'" -I{} bash -c "run_one \"\$@\"" _ {}
      '
  elif command -v docker >/dev/null 2>&1; then
    log "OpenFace via docker image: $OPENFACE_IMAGE (JOBS=$JOBS)"
    docker pull "$OPENFACE_IMAGE"
    # Single container, internal parallel loop. Repo is mounted at /work.
    docker run --rm -v "$WORKDIR":/work -w /work "$OPENFACE_IMAGE" bash -lc '
      set -e
      OF_BIN="$(command -v FeatureExtraction || echo /home/openface-build/build/bin/FeatureExtraction)"
      RAW_OUT="'"$raw_out"'"; OUT="'"$OPENFACE_DIR"'"; CLIPS="'"$CLIPS_TXT"'"; J="'"$JOBS"'"
      run_one() {
        local clip="$1"; local stem; stem="$(basename "${clip%.*}")"
        [[ -f "$OUT/${stem}_person0.csv" ]] && return 0
        "$OF_BIN" -f "$clip" -out_dir "$RAW_OUT" -q >/dev/null 2>&1 || { echo "FAIL $clip" >&2; return 0; }
        [[ -f "$RAW_OUT/${stem}.csv" ]] && mv -f "$RAW_OUT/${stem}.csv" "$OUT/${stem}_person0.csv"
        rm -f "$RAW_OUT/${stem}_of_details.txt"
      }
      export -f run_one
      export OF_BIN RAW_OUT OUT
      cat "$CLIPS" | xargs -P "$J" -I{} bash -c "run_one \"\$@\"" _ {}
    '
  else
    die "Neither a FeatureExtraction binary nor docker is available for OpenFace.
Tip: rent the vast.ai instance using the '$OPENFACE_IMAGE' image, or enable docker."
  fi

  rm -rf "$raw_out"
  local n; n="$(find "$OPENFACE_DIR" -name '*_person0.csv' | wc -l)"
  log "OpenFace CSVs written: $n -> $OPENFACE_DIR"
}

# -------------------------------- stage: i3d ----------------------------------
# Isolated env so mmaction's pins don't fight the base/training env. Mirrors
# scripts/extract_feature_cmd.txt (the CMOSE private path).
stage_i3d() {
  cd "$WORKDIR"
  [[ -f "$CLIPS_TXT" ]] || build_clips_txt
  mkdir -p "$I3D_DIR" artifacts/i3d

  if [[ ! -d .venv-i3d ]]; then
    log "Creating I3D env (.venv-i3d) + mmaction2"
    python3 -m venv .venv-i3d
    local pip=".venv-i3d/bin/python -m pip"
    $pip install -U pip setuptools wheel
    $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    $pip install mmengine mmcv-lite==2.1.0 opencv-python opencv-contrib-python decord av numpy
    $pip install --force-reinstall "git+https://github.com/open-mmlab/mmaction2.git@v1.2.0"
    # Patch packaging bug: localizers/__init__.py imports a DRN module absent here.
    .venv-i3d/bin/python - <<'PY'
import mmaction, pathlib
init = pathlib.Path(mmaction.__file__).parent / "models" / "localizers" / "__init__.py"
t = init.read_text(encoding="utf-8")
t = t.replace("from .drn.drn import DRN", "# from .drn.drn import DRN").replace(", 'DRN'", "")
init.write_text(t, encoding="utf-8")
print("patched", init)
PY
  fi
  local I3DPY="$WORKDIR/.venv-i3d/bin/python"

  if [[ ! -f artifacts/i3d/i3d_r50_kinetics400.pth ]]; then
    log "Downloading pretrained I3D checkpoint"
    curl -L --fail -o artifacts/i3d/i3d_r50_kinetics400.pth "$I3D_CKPT_URL"
  fi

  log "Building MMAction video list"
  awk '{print $0" 0"}' "$CLIPS_TXT" > artifacts/i3d/daisee_videos_for_mmaction.txt

  local config_py
  config_py="$("$I3DPY" - <<'PY'
import mmaction, pathlib
print(pathlib.Path(mmaction.__file__).parent / ".mim" / "configs" / "recognition" / "i3d" / "i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py")
PY
)"
  local extract_py
  extract_py="$("$I3DPY" - <<'PY'
import mmaction, pathlib
print(pathlib.Path(mmaction.__file__).parent / ".mim" / "tools" / "misc" / "clip_feature_extraction.py")
PY
)"

  log "Extracting clip-level I3D features"
  "$I3DPY" "$extract_py" \
    "$config_py" \
    artifacts/i3d/i3d_r50_kinetics400.pth \
    artifacts/i3d/raw_features \
    --video-list artifacts/i3d/daisee_videos_for_mmaction.txt \
    --cfg-options \
      test_dataloader.num_workers=4 \
      test_dataloader.persistent_workers=False \
      test_dataloader.dataset.data_prefix.video=''

  log "Converting I3D .pkl -> CMOSE .npy"
  "$WORKDIR/.venv/bin/python" scripts/daisee_convert_i3d.py \
    --raw-root artifacts/i3d/raw_features \
    --out-dir "$I3D_DIR"
  local n; n="$(find "$I3D_DIR" -name '*.npy' | wc -l)"
  log "I3D npy written: $n -> $I3D_DIR"
}

# -------------------------------- stage: push ---------------------------------
stage_push() {
  cd "$WORKDIR"

  # 1) Labels + scripts -> GitHub (small, safe).
  log "Committing labels + scripts to GitHub"
  [[ -n "${GIT_USER_EMAIL:-}" ]] && git config user.email "$GIT_USER_EMAIL"
  [[ -n "${GIT_USER_NAME:-}"  ]] && git config user.name  "$GIT_USER_NAME"
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    local host_path="${REPO_URL#https://}"
    git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@${host_path}"
  fi
  git add scripts/build_daisee_labels.py scripts/daisee_convert_i3d.py scripts/daisee_extract_vast.sh .gitignore \
          "$DAISEE_ROOT/final_data_1.json" "$DAISEE_ROOT/labels.csv" 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -m "Add DAiSEE engagement labels and extraction pipeline"
    git push origin "$REPO_BRANCH"
  else
    warn "Nothing new to commit"
  fi

  # 2) Features -> private Kaggle dataset (too big for git).
  if [[ -z "${KAGGLE_USERNAME:-}" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    warn "No Kaggle creds; skipping feature upload. Features remain at $OPENFACE_DIR and $I3D_DIR"
    return
  fi
  resolve_venv_py
  configure_kaggle
  log "Packaging features and creating private Kaggle dataset: $FEATURES_KAGGLE_SLUG"
  local up="$DAISEE_ROOT/_kaggle_upload"
  rm -rf "$up"; mkdir -p "$up"
  tar -czf "$up/openface_features.tar.gz" -C "$DAISEE_ROOT/features" openface
  tar -czf "$up/i3d_features.tar.gz"      -C "$DAISEE_ROOT/features" i3d
  cp "$DAISEE_ROOT/final_data_1.json" "$DAISEE_ROOT/labels.csv" "$up/"
  cat > "$up/dataset-metadata.json" <<JSON
{
  "title": "DAiSEE CMOSE-format features",
  "id": "$FEATURES_KAGGLE_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON
  if "$VENV_PY" -m kaggle datasets status "$FEATURES_KAGGLE_SLUG" >/dev/null 2>&1; then
    "$VENV_PY" -m kaggle datasets version -p "$up" -m "update $(date -u +%FT%TZ)" --dir-mode tar
  else
    "$VENV_PY" -m kaggle datasets create -p "$up" --dir-mode tar
  fi
  log "Uploaded features to https://www.kaggle.com/datasets/$FEATURES_KAGGLE_SLUG"
}

# --------------------------------- driver -------------------------------------
log "STAGES: $STAGES"
want repo     && stage_repo
want env      && stage_env
want kaggle   && stage_kaggle
want labels   && stage_labels
want openface && stage_openface
want i3d      && stage_i3d
want push     && stage_push

log "Done. Train on DAiSEE with:"
cat <<EOF
  cd "$WORKDIR" && .venv/bin/python main.py --model openface_tcn_i3d_fusion \\
    --labels_json   $DAISEE_ROOT/final_data_1.json \\
    --feature_dir   $OPENFACE_DIR \\
    --i3d_feature_dir $I3D_DIR
EOF
