#!/usr/bin/env bash
# install_eogs2.sh — set up EOGS++ (gardiens/EOGS2) on the 4090/WSL2 with our known CUDA fixes
# folded in (nvcc 12.1 on PATH, gcc-12 host compiler, RTX 4090 arch). Then run their install.sh,
# download the dataset, and prep affine cameras. Idempotent-ish. Run with sudo available.
#
#   bash scripts/eogs2/install_eogs2.sh
set -eo pipefail
EOGS2_DIR="${EOGS2_DIR:-$HOME/eogs-src/EOGS2}"
ENV="${EOGSPLUS_ENV:-eogsplus}"
PYV="3.9"

echo "==================== 0) clone EOGS2 ===================="
mkdir -p "$(dirname "$EOGS2_DIR")"
[ -d "$EOGS2_DIR/.git" ] || git clone https://github.com/gardiens/EOGS2.git "$EOGS2_DIR"
echo "EOGS2 @ $(git -C "$EOGS2_DIR" rev-parse --short HEAD)"

echo "==================== 1) CUDA toolchain (match torch cu12.1; 4090) ===================="
# their install.sh detects CUDA via `nvcc --version` and only accepts 11.8/12.0/12.1/11.6/11.4.
# Put the system CUDA 12.1 toolkit (installed for EOGS) on PATH so detection passes + kernels build.
for c in /usr/local/cuda-12.1 /usr/local/cuda; do
  if [ -x "$c/bin/nvcc" ] && "$c/bin/nvcc" --version 2>/dev/null | grep -q "release 12.1"; then
    export CUDA_HOME="$c"; export PATH="$c/bin:$PATH"; break
  fi
done
[ -n "${CUDA_HOME:-}" ] || { echo "!! no CUDA 12.1 nvcc found. Install it (see scripts/01_setup_env.sh) and re-run."; exit 1; }
command -v gcc-12 >/dev/null && export CC="$(command -v gcc-12)" && export CXX="$(command -v g++-12)"
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090
echo "CUDA_HOME=$CUDA_HOME  nvcc=$(nvcc --version | grep release)  CC=${CC:-default}"

echo "==================== 2) conda env '${ENV}' (python ${PYV}) ===================="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda install -n base conda-libmamba-solver -y >/dev/null 2>&1 || true
conda config --set solver libmamba || true
conda env list | grep -qE "^\s*${ENV}\s" || conda create -n "${ENV}" -c conda-forge "python=${PYV}" -y
conda activate "${ENV}"
python --version

echo "==================== 3) run EOGS2 install.sh (PyTorch + deps + CUDA kernels) ===================="
cd "$EOGS2_DIR"
# install.sh uses sudo apt-get for system libs; ensure available
bash install.sh || { echo "!! install.sh failed — paste the error; common cause is the kernel build (CUDA_HOME/gcc-12)."; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "==================== 4) dataset (PAN + MSI + pansharpen + rpc_ba/raw + truth) ===================="
if [ ! -d "$EOGS2_DIR/data/images" ]; then
  echo "downloading data.zip ..."
  wget -q -O /tmp/eogs2_data.zip "https://github.com/gardiens/EOGS2/releases/download/data/data.zip"
  unzip -q /tmp/eogs2_data.zip "data/*" -d "$EOGS2_DIR" && rm -f /tmp/eogs2_data.zip
else
  echo "data/ already present — skipping download."
fi

echo "==================== 5) affine camera prep ===================="
bash to_affine.sh || echo "   (to_affine.sh failed for some scenes — note which; we may need per-scene prep.)"

echo "==================== DONE ===================="
echo "Smoke test (one scene):"
echo "  conda activate ${ENV}; cd ${EOGS2_DIR}"
echo "  python src/gaussiansplatting/full_eval_pan.py experiments=eogsplus.yaml mode=3PAN rpc_type=rpc_ba scene=JAX_068"
