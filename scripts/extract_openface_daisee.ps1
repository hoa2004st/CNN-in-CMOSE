# Extract OpenFace features for all DAiSEE clips and build accepted.csv.
#
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\extract_openface_daisee.ps1
#
# Output (mirrors data/private):
#   data/DaiSEE/features/openface/<clip_stem>.csv
#   data/DaiSEE/accepted.csv
#   data/DaiSEE/accepted_clips.txt
#
# Already-processed clips are skipped; safe to re-run.

param(
    [string]$ClipsDir    = "data\DaiSEE\clips",
    [string]$OpenFaceDir = "data\DaiSEE\features\openface",
    [string]$OpenFaceBin = "third_party\OpenFace\build\bin",
    [string]$OpenCVBin   = "third_party\opencv\build\x64\vc16\bin",
    [string]$OpenBLASBin = "third_party\OpenFace\lib\3rdParty\OpenBLAS\bin\x64",
    [string]$VenvPy      = ".venv-vast-cu131\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Log($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }

$RepoRoot    = (Get-Item $PSScriptRoot).Parent.FullName
$ClipsDir    = Join-Path $RepoRoot $ClipsDir
$OpenFaceDir = Join-Path $RepoRoot $OpenFaceDir
$OpenFaceBin = Join-Path $RepoRoot $OpenFaceBin
$OpenCVBin   = Join-Path $RepoRoot $OpenCVBin
$OpenBLASBin = Join-Path $RepoRoot $OpenBLASBin
$VenvPy      = Join-Path $RepoRoot $VenvPy
$FExe        = Join-Path $OpenFaceBin "Release\FeatureExtraction.exe"

if (-not (Test-Path $FExe))    { throw "FeatureExtraction.exe not found at $FExe" }
if (-not (Test-Path $ClipsDir)){ throw "Clips dir not found: $ClipsDir  (run download_daisee.ps1 first)" }

$env:Path = "$OpenCVBin;$OpenBLASBin;$env:Path"
New-Item -ItemType Directory -Force -Path $OpenFaceDir | Out-Null

$allClips = Get-ChildItem -Path $ClipsDir -File | Where-Object { $_.Extension -in ".avi",".mp4",".mov" }
$total    = $allClips.Count
$done     = 0; $failed = 0; $skipped = 0

# Skip already-processed clips
$toProcess = $allClips | Where-Object { -not (Test-Path (Join-Path $OpenFaceDir ($_.BaseName + ".csv"))) }
$skipped   = $total - $toProcess.Count

Log "Found $total clips; $skipped already have CSVs; $($toProcess.Count) to process"
Log "Output -> $OpenFaceDir"

# Batch: pass 200 clips per FeatureExtraction.exe call so models load once per batch
$batchSize  = 200
$tmpDir     = Join-Path $RepoRoot "tmp\openface_daisee_tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$batchList  = [System.Collections.Generic.List[object]]::new()
foreach ($c in $toProcess) { $batchList.Add($c) }

$totalBatches = [math]::Ceiling($batchList.Count / $batchSize)
Log "Processing in $totalBatches batches of up to $batchSize clips each"

for ($b = 0; $b -lt $totalBatches; $b++) {
    $start      = $b * $batchSize
    $batchClips = $batchList.GetRange($start, [math]::Min($batchSize, $batchList.Count - $start))

    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

    $fArgs = @()
    foreach ($c in $batchClips) { $fArgs += "-f"; $fArgs += $c.FullName }
    $fArgs += "-out_dir"; $fArgs += $tmpDir; $fArgs += "-q"

    Push-Location $OpenFaceBin
    try { & ".\Release\FeatureExtraction.exe" @fArgs 2>$null } catch {}
    Pop-Location

    foreach ($c in $batchClips) {
        $rawCsv = Join-Path $tmpDir ($c.BaseName + ".csv")
        $outCsv = Join-Path $OpenFaceDir ($c.BaseName + ".csv")
        if (Test-Path $rawCsv) {
            Move-Item -Force $rawCsv $outCsv
            $done++
        } else {
            Warn "No output for $($c.Name)"
            $failed++
        }
    }

    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
    Log "Batch $($b+1)/$totalBatches done -- total: done=$done  failed=$failed  skipped=$skipped"
}

Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
Log "OpenFace extraction complete: done=$done  failed=$failed  skipped=$skipped"

# --- Build accepted.csv and accepted_clips.txt (pipe Python to stdin) ---
Log "Building accepted.csv"
$acceptedCsv = Join-Path $RepoRoot "data\DaiSEE\accepted.csv"
$acceptedTxt = Join-Path $RepoRoot "data\DaiSEE\accepted_clips.txt"

$pyCode = @"
import sys, pathlib
import pandas as pd

clips_dir    = pathlib.Path(r"$ClipsDir")
openface_dir = pathlib.Path(r"$OpenFaceDir")
out_csv      = pathlib.Path(r"$acceptedCsv")
out_txt      = pathlib.Path(r"$acceptedTxt")

clips = sorted(c for c in clips_dir.iterdir() if c.suffix.lower() in {'.avi', '.mp4', '.mov'})

rows = []
accepted_paths = []
for clip in clips:
    clip_id  = clip.stem
    csv_path = openface_dir / (clip_id + '.csv')
    ok, reason = 1, ''
    if not csv_path.exists():
        ok, reason = 0, 'openface_csv_missing'
    else:
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if len(df) == 0:
                ok, reason = 0, 'empty_csv'
            elif 'success' not in df.columns:
                ok, reason = 0, 'missing_success_col'
            else:
                succ  = int((df['success'] > 0.5).sum())
                ratio = succ / max(len(df), 1)
                if succ < 30:
                    ok, reason = 0, 'too_few_success_frames'
                elif ratio < 0.30:
                    ok, reason = 0, 'low_success_ratio'
        except Exception as e:
            ok, reason = 0, f'csv_parse_error: {e}'
    rows.append({'clip_id': clip_id, 'clip_path': str(clip), 'openface_csv': str(csv_path),
                 'is_accepted': ok, 'reject_reason': reason})
    if ok:
        accepted_paths.append(str(clip))

out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_csv, index=False)
out_txt.write_text('\n'.join(accepted_paths) + '\n', encoding='utf-8')
accepted = sum(r['is_accepted'] for r in rows)
print(f'accepted={accepted}  total={len(rows)}  -> {out_csv}')
"@

$pyCode | & $VenvPy -
if ($LASTEXITCODE -ne 0) { throw "Building accepted.csv failed" }

Log "Done. Next step: scripts\extract_i3d_daisee.ps1"
