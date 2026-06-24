#!/usr/bin/env bash
# 22_train_twosurface.sh — [Paper 1 / M8] train the GS-native two-surface model on a scene.
# Run 1: w_L_ground=0 (plumbing — DSM of the TOP must stay ~baseline).
# Run 2: w_L_ground=W (GEDI ground supervision). Live diagnostic prints |ground-GEDI| vs
# |top-GEDI| every 500 iters; the TOP DSM MAE is evaluated vs the airborne truth each run.
#
#   bash scripts/22_train_twosurface.sh [TILE] [W]      defaults JAX_113 1.0
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env" 2>/dev/null || true
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
EOGS_DIR="${EOGS_DIR:-$HOME/eogs-src/EOGS}"
TILE="${1:-JAX_113}"; W="${2:-1.0}"; MODE="${3:-ground}"; ITERS="${NUMITER:-5000}"   # MODE: ground|both|plumbing
ANCHORS="${ANCHORS:-${REPO_ROOT}/data/anchors/${TILE}_gedi_anchors.npz}"
[ -f "${ANCHORS}" ] || { echo "!! anchors missing: ${ANCHORS}"; exit 1; }

echo "==================== apply two-surface patch ===================="
python "${REPO_ROOT}/eogs_mods/apply_paper1_twosurface.py" "${EOGS_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate eogs
cd "${EOGS_DIR}/src/gaussiansplatting"
D="${EOGS_DIR}/data"; ts=$(date +%s)
run(){  # $1 tag  $2 w_ground
  local tag="$1" w="$2" exp="p1twosurf_${ts}_${TILE}_${tag}" out="${EOGS_DIR}/output/p1twosurf_${ts}_${TILE}_${tag}"
  echo "==================== TRAIN ${tag} (w_L_ground=${w}) ===================="
  python train.py -s "${D}/affine_models/${TILE}" --images "${D}/images/${TILE}" --eval \
    -m "${out}" --sh_degree 0 --iterations "${ITERS}" \
    --lidar_anchors_path "${ANCHORS}" --w_L_ground "${w}" --iterstart_L_ground 0 --w_L_groundtv "${WGTV:-0.1}"
  python render.py -m "${out}"
  local dsm; dsm=$(ls "${out}/test_opNone/ours_${ITERS}/dsm/" | sort -V | tail -1)
  echo "-------------------- ${tag}: TOP DSM MAE vs airborne truth --------------------"
  python "${EOGS_DIR}/scripts/eval/eval_dsm.py" \
    --pred-dsm-path "${out}/test_opNone/ours_${ITERS}/dsm/${dsm}" \
    --gt-dir "${D}/truth/${TILE}" --out-dir "${out}/" --aoi-id "${TILE}"
}
case "${MODE}" in
  plumbing) run plumbing 0 ;;
  both)     run plumbing 0; run ground "${W}" ;;
  *)        run ground "${W}" ;;   # default: ground only (plumbing already validated 1.32 m)
esac
echo
echo "READ: plumbing TOP DSM MAE should ~= baseline (${TILE} baseline 1.34 m) -> two-surface keeps the top."
echo "      In the 'ground' run, watch [P1-2surf] lines: |ground-GEDI| should fall well below |top-GEDI| (~canopy height)."
