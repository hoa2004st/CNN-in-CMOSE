# Vast.ai CUDA-like Environment (No Docker)

This project now includes a reproducible local setup that mirrors the NVIDIA CUDA-style Vast.ai template behavior on Windows.

## What was created

- `scripts/setup_vast_cuda_env.ps1`: idempotent setup script
- `requirements-vast-cu131.txt`: pinned base packages
- `requirements.lock.vast-cu131.txt`: exact frozen environment generated after install
- Virtual environment folder: `.venv-vast-cu131`

## Recreate the same environment

Run from project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_vast_cuda_env.ps1
```

Activate:

```powershell
.venv-vast-cu131\Scripts\Activate.ps1
```

## Verify GPU/CUDA

```powershell
nvidia-smi
.venv-vast-cu131\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected on this machine after setup:

- `torch 2.5.1+cu121`
- `torch.version.cuda = 12.1`
- `torch.cuda.is_available() = True`

## Notes

- Host NVIDIA driver controls runtime compatibility. Current host driver advertises CUDA `13.1`, which is compatible with the installed `cu121` PyTorch wheel.
- This is a functional match for CUDA development workflow, not a byte-for-byte match of Linux Docker internals.
