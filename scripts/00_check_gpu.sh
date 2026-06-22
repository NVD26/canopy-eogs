#!/usr/bin/env bash
# 00_check_gpu.sh — confirm the 4090 is visible and (if env exists) torch sees CUDA.
# Safe to run anytime. Run this FIRST.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

echo "==================== GPU / driver ===================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "!! nvidia-smi not found."
  echo "   On WSL2 you do NOT install Linux NVIDIA drivers — install the driver on"
  echo "   Windows; the GPU is then exposed to WSL2 automatically. Verify Windows"
  echo "   driver + WSL CUDA support, then re-run."
  exit 1
fi

echo
echo "==================== torch.cuda check ===================="
if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("torch built for CUDA:", torch.version.cuda)
PY
else
  echo "conda env '${CONDA_ENV}' not found yet — run scripts/01_setup_env.sh first."
fi
