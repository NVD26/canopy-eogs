#!/usr/bin/env bash
# 01_setup_env.sh — build the conda env, clone EOGS, install deps + CUDA kernels.
# Auto-installs Miniconda if missing, accepts conda ToS, installs system build
# tools, the conda env, PyTorch, EOGS, geospatial deps, a torch-matched CUDA
# toolkit, and the 3DGS CUDA kernels. Idempotent-ish: safe to re-run.
#
# Prereqs on the 4090 / WSL2:
#   - NVIDIA driver on Windows (exposes GPU to WSL2) — verify with scripts/00_check_gpu.sh
#   - sudo access (apt build tools + system CUDA toolkit on Debian/Ubuntu)
#
# NOTE: `set -eo pipefail` WITHOUT `-u` (nounset). conda activate/deactivate hooks
# (e.g. gdal's geotiff hook) reference unbound vars and would abort under `set -u`.
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
        build-essential cmake make gcc g++ gcc-12 g++-12 pkg-config \
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

echo "==================== 6) CUDA toolkit (matched to torch) + 3DGS CUDA kernels ===================="
# torch CUDA extensions need an nvcc whose CUDA version EXACTLY matches the one
# torch was built with (torch.version.cuda). A mismatched nvcc — from a system OR
# CONDA CUDA 13.x in the env — triggers "detected CUDA version mismatches PyTorch".
# Strategy: find a version-matched nvcc (prefer /usr/local/cuda-<ver>); if none,
# install the system CUDA toolkit for that version via NVIDIA's WSL2 apt repo;
# then build against it via CUDA_HOME. Any mismatched conda nvcc is ignored (it's
# harmless at runtime — torch ships its own CUDA libs).
CUDA_VER="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
[ -z "${CUDA_VER}" ] && CUDA_VER="12.1"
echo "torch was built with CUDA ${CUDA_VER}; locating a matching nvcc ..."

find_matched_nvcc() {
  local ver="$1" c
  for c in "/usr/local/cuda-${ver}/bin/nvcc" "/usr/local/cuda/bin/nvcc" \
           "${CONDA_PREFIX}/bin/nvcc" "$(command -v nvcc 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ] && "$c" --version 2>/dev/null | grep -q "release ${ver}"; then
      echo "$c"; return 0
    fi
  done
  return 1
}

NVCC="$(find_matched_nvcc "${CUDA_VER}" || true)"
if [ -z "${NVCC}" ] && command -v apt-get >/dev/null 2>&1; then
  echo "No CUDA ${CUDA_VER} nvcc found — installing system cuda-toolkit-${CUDA_VER//./-} (WSL2 apt; needs sudo)..."
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  if wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
     && ${SUDO} dpkg -i /tmp/cuda-keyring.deb \
     && ${SUDO} apt-get update -y \
     && ${SUDO} apt-get install -y "cuda-toolkit-${CUDA_VER//./-}"; then
    NVCC="$(find_matched_nvcc "${CUDA_VER}" || true)"
  else
    echo "   !! system CUDA toolkit install failed."
  fi
fi

if [ -n "${NVCC}" ]; then
  export CUDA_HOME="$(dirname "$(dirname "${NVCC}")")"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  # NB: do NOT name this 'CC' — that is the C-compiler env var; setting it to the
  # compute capability makes nvcc use "8.9" as the host compiler (-ccbin 8.9).
  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  [ -n "${COMPUTE_CAP}" ] && export TORCH_CUDA_ARCH_LIST="${COMPUTE_CAP}"   # RTX 4090 = 8.9
  # CUDA 12.x nvcc requires host gcc <= 12; the conda env often sets CC/CXX to a
  # newer gcc. Force gcc-12 for the kernel build if available (CC/CXX = real
  # compiler paths — nvcc uses CC for -ccbin).
  if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
    export CC="$(command -v gcc-12)"; export CXX="$(command -v g++-12)"
    echo "Host compiler for kernels: ${CC} / ${CXX}"
  fi
  echo "Using nvcc: ${NVCC}"
  "${NVCC}" --version | tail -1
  echo "CUDA_HOME=${CUDA_HOME}  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-auto}"
  # --no-build-isolation so the build sees torch; --no-cache-dir avoids reusing a failed build.
  pip install --no-build-isolation --no-cache-dir "${EOGS_DIR}/src/gaussiansplatting/submodules/diff-gaussian-rasterization"
  pip install --no-build-isolation --no-cache-dir "${EOGS_DIR}/src/gaussiansplatting/submodules/simple-knn"
  echo "CUDA kernels built."
else
  echo "!! Could not obtain a CUDA ${CUDA_VER} nvcc. Install it manually then re-run:"
  echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb"
  echo "   sudo dpkg -i /tmp/ck.deb && sudo apt-get update && sudo apt-get install -y cuda-toolkit-${CUDA_VER//./-}"
fi

echo
echo "==================== DONE ===================="
echo "Next: conda activate ${CONDA_ENV} && bash scripts/run_milestone.sh"
echo "Record the EOGS commit above in STATUS.md §7."
