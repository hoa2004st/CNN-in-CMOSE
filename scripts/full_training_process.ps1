# Run the full model x loss training sweep in one Python process.
#
# Trains every comparison model (openface_mlp, temporal_cnn, lstm, transformer,
# i3d_mlp) across the three losses (cross_entropy, weighted_cross_entropy,
# ordinal) and writes runs under <RunRoot>/<model>/<loss>/.
#
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\full_training_process.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\full_training_process.ps1 -Device cuda -Epochs 400

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
$LogPath = Join-Path $LogDir "full_training_process.log"

Write-Host "Run root : $RunRoot" -ForegroundColor Cyan
Write-Host "Dataset  : $Dataset" -ForegroundColor Cyan
Write-Host "Device   : $Device"  -ForegroundColor Cyan
Write-Host "Epochs   : $Epochs"  -ForegroundColor Cyan

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
    Tee-Object -FilePath $LogPath -Append

if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }

Write-Host "All training runs completed." -ForegroundColor Green
