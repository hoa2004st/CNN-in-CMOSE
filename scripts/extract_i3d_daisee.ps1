# Set up the MMAction2 venv and extract I3D features for accepted DAiSEE clips.
#
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\extract_i3d_daisee.ps1
#
# Output (mirrors data/private):
#   data/DaiSEE/features/i3d/<clip_stem>.npy
#
# Re-running is safe: already-converted .npy files are skipped.

param(
    [string]$AcceptedTxt  = "data\DaiSEE\accepted_clips.txt",
    [string]$I3DDir       = "data\DaiSEE\features\i3d",
    [string]$VenvI3D      = ".venv-i3d",
    [string]$VenvPy       = ".venv-vast-cu131\Scripts\python.exe",
    [string]$ArtifactsDir = "artifacts\i3d",
    [string]$I3DCheckpointUrl = "https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Log($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }

$RepoRoot    = (Get-Item $PSScriptRoot).Parent.FullName
$AcceptedTxt = Join-Path $RepoRoot $AcceptedTxt
$I3DDir      = Join-Path $RepoRoot $I3DDir
$VenvI3D     = Join-Path $RepoRoot $VenvI3D
$ArtDir      = Join-Path $RepoRoot $ArtifactsDir
$I3DPy       = Join-Path $VenvI3D "Scripts\python.exe"
$VenvPy      = Join-Path $RepoRoot $VenvPy

if (-not (Test-Path $AcceptedTxt)) {
    throw "$AcceptedTxt not found -- run extract_openface_daisee.ps1 first"
}

# --- 1. Create .venv-i3d if missing ---
if (-not (Test-Path $I3DPy)) {
    Log "Creating .venv-i3d"
    python -m venv $VenvI3D
    & $I3DPy -m pip install -U pip setuptools wheel
    Log "Installing PyTorch (CUDA 12.1)"
    & $I3DPy -m pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    Log "Installing MMAction2 dependencies"
    & $I3DPy -m pip install mmengine "mmcv-lite==2.1.0" opencv-python opencv-contrib-python decord av numpy importlib_metadata
    Log "Installing mmaction2 v1.2.0 (this takes a few minutes)"
    & $I3DPy -m pip install --force-reinstall "git+https://github.com/open-mmlab/mmaction2.git@v1.2.0"
    if ($LASTEXITCODE -ne 0) { throw "mmaction2 install failed" }

    # Patch DRN import bug
    $patchCode = @"
from pathlib import Path
import mmaction
p = Path(mmaction.__file__).parent / 'models' / 'localizers' / '__init__.py'
t = p.read_text(encoding='utf-8')
t = t.replace('from .drn.drn import DRN', '# from .drn.drn import DRN')
t = t.replace(", 'DRN'", '')
p.write_text(t, encoding='utf-8')
print('patched', p)
"@
    $patchCode | & $I3DPy -
    if ($LASTEXITCODE -ne 0) { throw "mmaction2 patch failed" }
} else {
    Log ".venv-i3d already present -- skipping setup"
}

# --- 2. Download I3D checkpoint ---
New-Item -ItemType Directory -Force -Path $ArtDir | Out-Null
$ckpt = Join-Path $ArtDir "i3d_r50_kinetics400.pth"
if (-not (Test-Path $ckpt)) {
    Log "Downloading I3D checkpoint (~180 MB)"
    Invoke-WebRequest -Uri $I3DCheckpointUrl -OutFile $ckpt
} else {
    Log "I3D checkpoint already present"
}

# --- 3. Build MMAction video list ---
$videoListFile = Join-Path $ArtDir "daisee_accepted_for_mmaction.txt"
Log "Building MMAction video list from $AcceptedTxt"
$lines = Get-Content $AcceptedTxt | Where-Object { $_.Trim() -ne "" }
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllLines($videoListFile, ($lines | ForEach-Object { "$_ 0" }), $utf8NoBom)
Log "Video list: $($lines.Count) clips -> $videoListFile"

# --- 4. Locate mmaction config and extraction tool ---
$mmConfig = (& $I3DPy -c "import mmaction, pathlib; print(pathlib.Path(mmaction.__file__).parent / '.mim' / 'configs' / 'recognition' / 'i3d' / 'i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py')").Trim()
$mmTool   = (& $I3DPy -c "import mmaction, pathlib; print(pathlib.Path(mmaction.__file__).parent / '.mim' / 'tools' / 'misc' / 'clip_feature_extraction.py')").Trim()

$rawFeatDir = Join-Path $ArtDir "raw_features"
New-Item -ItemType Directory -Force -Path $rawFeatDir | Out-Null

# --- 5. Extract I3D features (GPU) ---
Log "Extracting clip-level I3D features (GPU)"
& $I3DPy $mmTool $mmConfig $ckpt $rawFeatDir `
    --video-list $videoListFile `
    --cfg-options `
        test_dataloader.num_workers=0 `
        test_dataloader.persistent_workers=False `
        "test_dataloader.dataset.data_prefix.video=''"

if ($LASTEXITCODE -ne 0) { throw "I3D feature extraction failed" }

# --- 6. Convert .pkl -> .npy ---
# clip_feature_extraction.py's split_feats() writes per-clip pkls next to the
# source videos (osp.join with absolute video paths discards the out-dir prefix).
Log "Converting .pkl -> .npy"
New-Item -ItemType Directory -Force -Path $I3DDir | Out-Null
& $VenvPy (Join-Path $RepoRoot "scripts\daisee_convert_i3d.py") `
    --raw-root $ClipsDir `
    --out-dir  $I3DDir

if ($LASTEXITCODE -ne 0) { throw "I3D conversion failed" }

# Clean up per-clip pkls from clips dir
Get-ChildItem -Path $ClipsDir -Filter "*.pkl" -ErrorAction SilentlyContinue | Remove-Item -Force

$count = (Get-ChildItem -Path $I3DDir -Filter "*.npy" -ErrorAction SilentlyContinue).Count
Log "I3D .npy files written: $count -> $I3DDir"
Log "Done. Next steps:"
Log "  python scripts\build_daisee_labels.py --dataset-root data\DaiSEE\raw"
Log "  python scripts\compare_naive_models.py --feature_dir data/DaiSEE/features/openface --labels_json data/DaiSEE/final_data_1.json --i3d_feature_dir data/DaiSEE/features/i3d --run_root outputs/training_log/daisee --amp"
