# Download DAiSEE from Kaggle and flatten all video clips into data/DaiSEE/clips/.
#
# Prerequisites:
#   1. Fill in kaggle.json at the repo root with your username and API key.
#   2. Run from the repo root:
#        powershell -ExecutionPolicy Bypass -File scripts\download_daisee.ps1

param(
    [string]$VenvPy        = ".venv-vast-cu131\Scripts\python.exe",
    [string]$KaggleDataset = "olgaparfenova/daisee",
    [string]$RawDir        = "data\DaiSEE\raw",
    [string]$ClipsDir      = "data\DaiSEE\clips"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Log($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }

$RepoRoot  = (Get-Item $PSScriptRoot).Parent.FullName
$RawDir    = Join-Path $RepoRoot $RawDir
$ClipsDir  = Join-Path $RepoRoot $ClipsDir
$VenvPy    = Join-Path $RepoRoot $VenvPy

# --- 1. Kaggle credentials ---
$kaggleJsonSrc = Join-Path $RepoRoot "kaggle.json"
if (-not (Test-Path $kaggleJsonSrc)) {
    throw "kaggle.json not found at $kaggleJsonSrc"
}
$cred = Get-Content $kaggleJsonSrc -Raw | ConvertFrom-Json
if ($cred.username -eq "YOUR_KAGGLE_USERNAME") {
    throw "Edit kaggle.json and replace the placeholder username and key."
}
$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null
Copy-Item -Force $kaggleJsonSrc (Join-Path $kaggleDir "kaggle.json")
Log "Kaggle credentials -> $kaggleDir\kaggle.json"

# --- 2. Install kaggle package ---
Log "Ensuring kaggle package is installed"
& $VenvPy -m pip install --upgrade kaggle
if ($LASTEXITCODE -ne 0) { throw "pip install kaggle failed" }
$KaggleCLI = Join-Path (Split-Path $VenvPy) "kaggle.exe"

# --- 3. Download ---
New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
$existingAvi = Get-ChildItem -Path $RawDir -Recurse -Include "*.avi","*.mp4" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existingAvi) {
    Warn "Video files already present under $RawDir -- skipping download"
} else {
    Log "Downloading $KaggleDataset (~14 GB) into $RawDir"
    & $KaggleCLI datasets download -d $KaggleDataset -p $RawDir
    if ($LASTEXITCODE -ne 0) { throw "kaggle download failed" }

    Log "Extracting ZIP archive(s)"
    $zips = Get-ChildItem -Path $RawDir -Filter "*.zip"
    foreach ($zip in $zips) {
        Log "Extracting $($zip.Name) ..."
        Expand-Archive -Path $zip.FullName -DestinationPath $RawDir -Force
        Remove-Item -Force $zip.FullName
        Log "Extracted and removed $($zip.Name)"
    }
}

# --- 4. Flatten clips into data/DaiSEE/clips/ ---
New-Item -ItemType Directory -Force -Path $ClipsDir | Out-Null
$allVideos = Get-ChildItem -Path $RawDir -Recurse -Include "*.avi","*.mp4" -ErrorAction SilentlyContinue
Log "Found $($allVideos.Count) video clips -- flattening into $ClipsDir"

$copied  = 0
$skipped = 0
foreach ($v in $allVideos) {
    $dest = Join-Path $ClipsDir $v.Name
    if (Test-Path $dest) {
        $skipped++
    } else {
        Copy-Item -Force $v.FullName $dest
        $copied++
    }
    if ((($copied + $skipped) % 500) -eq 0 -and ($copied + $skipped) -gt 0) {
        Log "  copied=$copied  skipped=$skipped"
    }
}
Log "Clips ready: copied=$copied  already_existed=$skipped  total=$(($copied + $skipped))"
Log "Done. Next step: python scripts\build_daisee_labels.py --dataset-root $RawDir"
