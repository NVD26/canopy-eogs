#!/usr/bin/env bash
# 21_build_tile_scene.sh — [Paper 1 / M8] build ONE EOGS scene from a DFC2019 JAX tile, using
# Sat-NeRF's create_satellite_dataset (sun angles from the metadata server; original RPCs first,
# BA added later once validated). Stages outputs into the EOGS data tree + runs to_affine.
#
#   bash scripts/21_build_tile_scene.sh JAX_113
# env: conda 'ba' (has bundle_adjust, rasterio, fire). EOGS clone at $EOGS_DIR.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env" 2>/dev/null || true
EOGS_DIR="${EOGS_DIR:-$HOME/eogs-src/EOGS}"
DFC="${DFC2019_DIR:-$HOME/eogs-src/DFC2019}"
SATNERF="${TOOLS_DIR:-$HOME/eogs-src/tools}/satnerf"
SCENES="${SCENES_DIR:-$HOME/eogs-src/scenes}"
TILE="${1:-JAX_113}"
RGBSRC="${DFC}/Track3-RGB-1"          # JAX RGB crops live here (flat, JAX_<tile>_<view>_RGB.tif)

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ba
python -c "import fire" 2>/dev/null || python -m pip install fire

echo "==================== 1) group flat RGB crops into Track3-RGB/${TILE}/ ===================="
mkdir -p "${DFC}/Track3-RGB/${TILE}"
n=0; for f in "${RGBSRC}/${TILE}"_*_RGB.tif; do
  [ -e "$f" ] || continue; case "$f" in *Zone.Identifier) continue;; esac
  ln -sf "$f" "${DFC}/Track3-RGB/${TILE}/$(basename "$f")"; n=$((n+1))
done
echo "  linked ${n} views for ${TILE}"
[ "${n}" -gt 0 ] || { echo "!! no RGB views found for ${TILE} in ${RGBSRC}"; exit 1; }

echo "==================== 2) create_satellite_dataset (original RPCs, no re-crop) ===================="
mkdir -p "${SCENES}"
cd "${SATNERF}"
# ba=False (validate first; BA added in a follow-up), crop_aoi=False (use the DFC crops as-is)
python create_satellite_dataset.py "${TILE}" "${DFC}" "${SCENES}/${TILE}" --ba False --crop_aoi False --splits True

echo "==================== 3) stage into the EOGS data tree ===================="
mkdir -p "${EOGS_DIR}/data/rpcs/${TILE}" "${EOGS_DIR}/data/images/${TILE}" "${EOGS_DIR}/data/truth/${TILE}"
cp "${SCENES}/${TILE}"/*.json "${EOGS_DIR}/data/rpcs/${TILE}/"
cp "${SCENES}/${TILE}"/train.txt "${SCENES}/${TILE}"/test.txt "${EOGS_DIR}/data/rpcs/${TILE}/" 2>/dev/null || true
for f in "${DFC}/Track3-RGB/${TILE}"/*.tif; do cp "$(readlink -f "$f")" "${EOGS_DIR}/data/images/${TILE}/$(basename "$f")"; done
cp "${DFC}/Track3-Truth/${TILE}_DSM.tif" "${DFC}/Track3-Truth/${TILE}_DSM.txt" "${EOGS_DIR}/data/truth/${TILE}/" 2>/dev/null || true
cp "${DFC}/Track3-Truth/${TILE}_CLS.tif" "${EOGS_DIR}/data/truth/${TILE}/" 2>/dev/null || true

echo "==================== 4) to_affine (allow new tiles) ===================="
TOAFF="${EOGS_DIR}/src/gaussiansplatting/scripts/dataset_creation/to_affine.py"
[ -f "${TOAFF}" ] || TOAFF="${EOGS_DIR}/scripts/dataset_creation/to_affine.py"
# remove the hard-coded 7-scene assert so arbitrary tiles work (idempotent)
python - "$TOAFF" <<'PY'
import sys,re
p=sys.argv[1]; s=open(p).read()
s2=re.sub(r'\n\s*assert scene_name in \[[^\]]*\]\n', '\n', s, count=1)
if s2!=s: open(p,'w').write(s2); print("  patched: removed scene_name assert")
else: print("  (assert already absent)")
PY
conda activate eogs
cd "${EOGS_DIR}"
TOAFF_REL="${TOAFF#${EOGS_DIR}/}"
python "${TOAFF_REL}" --root_dir data/rpcs --scene_name "${TILE}"

echo "==================== DONE: ${TILE} scene built ===================="
echo "  affine -> ${EOGS_DIR}/data/affine_models/${TILE}/affine_models.json"
echo "  images -> ${EOGS_DIR}/data/images/${TILE}/   truth -> ${EOGS_DIR}/data/truth/${TILE}/"
echo "  Next: train with  bash scripts/15_train_lidar.sh ${TILE} 0   (baseline, then M8)"
