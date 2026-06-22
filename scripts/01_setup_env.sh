#!/usr/bin/env bash
# 01_setup_env.sh — build the conda env, clone EOGS, install deps + CUDA kernels.
# Idempotent-ish: skips steps already done. Run once per machine.
#
# Prereqs on the 4090 / WSL2:
#   - conda / miniconda installed
#   - NVIDIA driver on Windows (exposes GPU to WSL2) — verify with scripts/00_check_gpu.sh
#   - CUDA toolkit + nvcc available for building the 3DGS CUDA kernels.
#       On WSL2: install the CUDA toolkit (e.g. `sudo apt install cuda-toolkit-12-1`
#       or NVIDIA's WSL-Ubuntu toolkit) and ensure `nvcc --version` works and
#       CUDA_HOME points at it. The torch cudatoolkit alone is NOT enough to
#       compile diff-gaussian-rasterization.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

echo "==================== 1) conda env: ${CONDA_ENV} ===================="
if ! command -v conda >/dev/null 2>&1; then
  echo "!! conda not found. Install Miniconda first."; exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  echo "env '${CONDA_ENV}' already exists — reusing."
else
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"
python --version

echo "==================== 2) PyTorch (${TORCH_INDEX_URL}) ===================="
pip install --upgrade pip setuptools wheel packaging
pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "==================== 3) clone EOGS (recursive) ===================="
if [ -d "${EOGS_DIR}/.git" ]; then
  echo "EOGS already cloned at ${EOGS_DIR} — fetching."
  git -C "${EOGS_DIR}" fetch --all --recurse-submodules || true
else
  mkdir -p "$(dirname "${EOGS_DIR}")"
  git clone --recursive "${EOGS_REPO_URL}" "${EOGS_DIR}"
fi
if [ -n "${EOGS_COMMIT}" ]; then
  git -C "${EOGS_DIR}" checkout "${EOGS_COMMIT}"
  git -C "${EOGS_DIR}" submodule update --init --recursive
fi
echo "EOGS commit: $(git -C "${EOGS_DIR}" rev-parse HEAD)"

echo "==================== 4) EOGS python requirements ===================="
pip install -r "${EOGS_DIR}/requirements.txt"

echo "==================== 5) geospatial + lidar tooling (for Paper 1 later) ===================="
# gdal can be fussy via pip; conda-forge is more reliable. Try conda first, fall back to pip.
conda install -n "${CONDA_ENV}" -c conda-forge gdal -y || pip install gdal || \
  echo "   (gdal optional for the milestone; revisit before Paper 1 lidar work)"
pip install rasterio rpcm pyproj laspy h5py shapely earthaccess

echo "==================== 6) build 3DGS CUDA kernels ===================="
# These compile against your CUDA toolkit; needs nvcc + CUDA_HOME.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "!! nvcc not found — cannot build CUDA kernels. Install the CUDA toolkit"
  echo "   (see header notes), then re-run this script. Skipping for now."
else
  echo "nvcc: $(nvcc --version | tail -1)"
  pip install "${EOGS_DIR}/src/gaussiansplatting/submodules/diff-gaussian-rasterization"
  pip install "${EOGS_DIR}/src/gaussiansplatting/submodules/simple-knn"
fi

echo
echo "==================== DONE ===================="
echo "Next: conda activate ${CONDA_ENV} && bash scripts/run_milestone.sh"
echo "Record the EOGS commit above in STATUS.md §7."
