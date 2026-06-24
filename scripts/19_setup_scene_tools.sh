#!/usr/bin/env bash
# 19_setup_scene_tools.sh — [Paper 1 / M8 stage 1] set up the tooling to BUILD a larger
# EOGS scene from DFC2019: Sat-NeRF's dataset creator + bundle_adjust (accurate RPCs).
# Folds-in fixes as we hit them (advisor reproducibility rule). Best-effort; iterate on errors.
#
#   bash scripts/19_setup_scene_tools.sh
# Creates a conda env 'ba', clones the tools, and prints the DFC2019 data download steps.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env" 2>/dev/null || true
TOOLS="${TOOLS_DIR:-$HOME/eogs-src/tools}"
DATA="${DFC2019_DIR:-$HOME/eogs-src/DFC2019}"
mkdir -p "${TOOLS}" "${DATA}"

echo "==================== 1) clone scene-build tools ===================="
clone() { [ -d "$2/.git" ] || git clone --depth 1 "$1" "$2"; echo "  $2 @ $(git -C "$2" rev-parse --short HEAD 2>/dev/null || echo '?')"; }
clone https://github.com/centreborelli/satnerf.git          "${TOOLS}/satnerf"
clone https://github.com/centreborelli/sat-bundleadjust.git "${TOOLS}/sat-bundleadjust"

echo "==================== 2) conda env 'ba' (bundle_adjust + s2p deps) ===================="
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^\s*ba\s"; then
  conda create -n ba -c conda-forge "python=3.9" pip -y
fi
conda activate ba
# the env may pre-exist without pip (earlier run) — guarantee pip either way
python -m ensurepip --upgrade 2>/dev/null || conda install -n ba -c conda-forge pip -y
python -m pip install --upgrade pip setuptools wheel
# heavy geospatial libs via conda-forge (binary wheels; avoids source-build failures)
conda install -n ba -c conda-forge -y \
  numpy scipy matplotlib rasterio pyproj gdal opencv numba affine shapely libgdal
# pure-python / photogrammetry deps via pip
python -m pip install rpcm srtm4 utm plyfile || \
  echo "  !! some pip deps failed — note which and we fold the fix in here."
# bundle_adjust package (from the cloned repo)
python -m pip install --no-build-isolation -e "${TOOLS}/sat-bundleadjust" || \
  echo "  !! bundle_adjust editable install failed — check its README deps, fold fix in here."
# (s2p intentionally omitted: classic-MVS comparison only, pins an unbuildable pyproj; not needed to BUILD the scene.)

echo "---- verify bundle_adjust imports ----"
python -c "import bundle_adjust; print('bundle_adjust OK')" || echo "  !! bundle_adjust import failed — paste the error."

echo "==================== 3) DFC2019 source data (manual/open) ===================="
cat <<'TXT'
  Need (open access): DFC2019 'Track3-RGB' (multi-view WV3 RGB + RPC) and 'Track3-Truth'
  (airborne DSM) for Jacksonville. Put them under: ${DFC2019_DIR:-~/eogs-src/DFC2019}/
    DFC2019/Track3-RGB/    DFC2019/Track3-Truth/
  Sources:
    - IEEE DataPort: https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
    - Baseline/help: https://github.com/pubgeo/dfc2019
  (Track3-RGB is the larger source imagery the create script crops per-AOI. Several GB.)
TXT

echo "DONE (stage 1). Next: scripts/20 will crop our ~1 km AOI + bundle-adjust into an EOGS scene."
