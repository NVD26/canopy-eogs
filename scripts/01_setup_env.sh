#!/usr/bin/env bash
# 01_setup_env.sh — build the conda env, clone EOGS, install deps + CUDA kernels.
# Auto-installs Miniconda if missing, accepts conda ToS, installs system build
# tools, then EOGS + geospatial deps. Idempotent-ish: safe to re-run.
#
# Prereqs on the 4090 / WSL2:
#   - NVIDIA driver on Windows (exposes GPU to WSL2) — verify with scripts/00_check_gpu.sh
#   - sudo access (for the apt build-tools step on Debian/Ubuntu)
#   - CUDA toolkit + nvcc for building the 3DGS CUDA kernels (step 6). The torch
#     cudatoolkit alone is NOT enough to compile diff-gaussian-rasterization.
#
# NOTE: we intentionally use `set -eo pipefail` WITHOUT `-u` (nounset). conda's
# activate/deactivate hook scripts (e.g. gdal's geotiff-deactivate.sh) reference
# unbound vars like _CONDA_SET_GEOTIFF_CSV; under `set -u` that aborts the script
# with "unbound variable". Dropping -u keeps conda activation working.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

echo "==================== 0) ensure conda is available ===================="
if ! command -v conda >/dev/null 2>&1; then
  if [ -x "${HOME}/miniconda3/bin/conda" ]; then
    echo "Found existing Miniconda at ~/miniconda3 — using it."
  else
    echo "conda not found — installing Miniconda to ~/miniconda3 ..."
    ARCH="$(uname -m)"
    case "${ARCH}" in
      x86_64)        MC_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
      aarch64|arm64) MC_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
      *) echo "!! Unsupported arch '${ARCH}'. Install Miniconda manually, then re-run."; exit 1 ;;
    esac
    MC_INSTALLER="${HOME}/miniconda_installer.sh"
    if command -v wget >/dev/null 2>&1; then
      wget -q -O "${MC_INSTALLER}" "${MC_URL}"
    else
      curl -fsSL -o "${MC_INSTALLER}" "${MC_URL}"
    fi
    bash "${MC_INSTALLER}" -b -p "${HOME}/miniconda3"
    rm -f "${MC_INSTALLER}"
    "${HOME}/miniconda3/bin/conda" init bash || true
    echo "Miniconda installed to ~/miniconda3 (run 'source ~/.bashrc' later for interactive use)."
  fi
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
echo "conda: $(command -v conda)"

echo "==================== 0b) accept Anaconda default-channel ToS (best-effort) ===================="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

echo "==================== 0c) system build tools (Debian/Ubuntu; needs sudo) ===================="
# EOGS deps (iio, plyflatten, srtm4) compile C/C++ — they need make, a compiler,
# and image/geo dev libraries. Without these you get '/bin/sh: make: not found'.
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  ${SUDO} apt-get update -y || true
  if ! ${SUDO} apt-get install -y \
        build-essential cmake make gcc g++ pkg-config \
        libgdal-dev gdal-bin libtiff-dev libpng-dev libjpeg-dev libfftw3-dev \
        git wget unzip; then
    echo "   !! apt install failed. Run this manually with sudo, then re-run the script:"
    echo "      sudo apt-get install -y build-essential cmake libgdal-dev gdal-bin \\"
    echo "           libtiff-dev libpng-dev libjpeg-dev libfftw3-dev"
  fi
else
  echo "   apt-get not found — install make/gcc/g++ and libtiff/png/jpeg/gdal dev libs yourself."
fi

echo "==================== 1) conda env: ${CONDA_ENV} ===================="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  echo "env '${CONDA_ENV}' already exists — reusing."
else
  conda create -n "${CONDA_ENV}" -c conda-forge "python=${PYTHON_VERSION}" -y
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
conda install -n "${CONDA_ENV}" -c conda-forge gdal -y || pip install gdal || \
  echo "   (gdal optional for the milestone; revisit before Paper 1 lidar work)"
pip install rasterio rpcm pyproj laspy h5py shapely earthaccess

echo "==================== 6) build 3DGS CUDA kernels ===================="
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
