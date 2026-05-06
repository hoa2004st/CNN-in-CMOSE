param(
    [string]$EnvName = ".venv-vast-cu131",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Run([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

Step "Checking NVIDIA driver and CUDA runtime"
Run { nvidia-smi }

Step "Checking Python"
Run { & $PythonExe --version }

if (-not (Test-Path $EnvName)) {
    Step "Creating virtual environment: $EnvName"
    & $PythonExe -m venv $EnvName
}
else {
    Step "Virtual environment already exists: $EnvName"
}

$VenvPython = Join-Path $EnvName "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python not found at $VenvPython"
}

Step "Upgrading pip/setuptools/wheel"
Run { & $VenvPython -m pip install --upgrade pip setuptools wheel }

Step "Installing pinned base packages"
Run { & $VenvPython -m pip install -r "requirements-vast-cu131.txt" }

Step "Installing PyTorch CUDA build"
Run { & $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 }

Step "Installing project requirements"
Run { & $VenvPython -m pip install -r "requirements.txt" }

Step "Verifying CUDA in torch"
Run { & $VenvPython -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" }

Step "Freezing exact environment"
Run { & $VenvPython -m pip freeze | Out-File -Encoding ascii "requirements.lock.vast-cu131.txt" }

Write-Host "`nEnvironment is ready." -ForegroundColor Green
Write-Host "Activate with: $EnvName\Scripts\Activate.ps1"
