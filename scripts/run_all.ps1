# End-to-end pipeline: train every model x loss run, then regenerate the
# per-clip CMOSE-test prediction table consumed by the analysis layer.
#
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Device cuda

param(
    [string]$RunRoot    = "outputs/training_log",
    [string]$Dataset    = "cmose",
    [int]   $Epochs     = 400,
    [int]   $BatchSize  = 64,
    [double]$Lr         = 1e-4,
    [int]   $Patience   = 10,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device     = "auto",
    [int]   $NumWorkers = 4
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot

$LogDir = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Run root : $RunRoot" -ForegroundColor Cyan
Write-Host "Device   : $Device"  -ForegroundColor Cyan

# 1) Train the full model x loss sweep.
python -m src.training.full_training_process `
    --run_root     $RunRoot `
    --dataset      $Dataset `
    --epochs       $Epochs `
    --batch_size   $BatchSize `
    --lr           $Lr `
    --patience     $Patience `
    --device       $Device `
    --num_workers  $NumWorkers `
    --naive_losses cross_entropy weighted_cross_entropy ordinal 2>&1 |
    Tee-Object -FilePath (Join-Path $LogDir "full_training_process.log") -Append
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }

# 2) Regenerate the per-clip prediction table.
python -m src.analysis.prediction_generator `
    --device     $Device `
    --batch_size $BatchSize 2>&1 |
    Tee-Object -FilePath (Join-Path $LogDir "prediction_generator.log") -Append
if ($LASTEXITCODE -ne 0) { throw "Prediction generation failed with exit code $LASTEXITCODE" }

Write-Host "Training and prediction pipeline completed." -ForegroundColor Green
