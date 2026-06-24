#!/usr/bin/env bash
# 16_lidar_weight_sweep.sh — [Paper 1 / M7 diagnostic] characterise the canopy-top
# lidar loss as a DOSE-RESPONSE: train JAX_068 at several w_L_lidar values (same scene
# & seed each time; only the weight changes), eval DSM MAE (overall + tree), tabulate.
# A monotonic trend confirms the loss is active and the canopy-top TARGET — not a bug —
# is what moves the error. Motivates M8 (ground / two-surface).
#
#   bash scripts/16_lidar_weight_sweep.sh [SCENE] ["w1 w2 ..."]
# defaults: SCENE=JAX_068  WEIGHTS="0 0.01 0.03 0.1 0.3"
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
EOGS_DIR="${EOGS_DIR:-$HOME/eogs-src/EOGS}"
SCENE="${1:-JAX_068}"
WEIGHTS="${2:-0 0.01 0.03 0.1 0.3}"
ITERS="${NUMITER:-5000}"
ANCHORS="${ANCHORS:-${REPO_ROOT}/data/anchors/${SCENE}_gedi_anchors.npz}"
[ -f "${ANCHORS}" ] || { echo "!! anchors missing: ${ANCHORS}"; exit 1; }

echo "==================== applying patch ===================="
python "${REPO_ROOT}/eogs_mods/apply_paper1_lidar.py" "${EOGS_DIR}"

cd "${EOGS_DIR}/src/gaussiansplatting"
data="${EOGS_DIR}/data"
ts=$(date +%s)
mkdir -p "${REPO_ROOT}/results"
SUMMARY="${REPO_ROOT}/results/lidar_weight_sweep_${SCENE}_${ts}.csv"
echo "scene,w_L_lidar,mae_overall,mae_tree" > "${SUMMARY}"

grab_mae () { grep -Eo 'MAE: *[0-9.]+' | tail -n1 | grep -Eo '[0-9.]+'; }

for w in ${WEIGHTS}; do
  exp="p1sweep_${ts}_${SCENE}_w${w}"
  out="${EOGS_DIR}/output/${exp}"
  echo "==================== TRAIN w_L_lidar=${w} ===================="
  python train.py -s "${data}/affine_models/${SCENE}" --images "${data}/images/${SCENE}" \
    --eval -m "${out}" --sh_degree 0 --iterations "${ITERS}" \
    --lidar_anchors_path "${ANCHORS}" --w_L_lidar "${w}" --iterstart_L_lidar 0
  python render.py -m "${out}"
  dsm=$(ls "${out}/test_opNone/ours_${ITERS}/dsm/" | sort -V | tail -n1)
  mo=$(python "${EOGS_DIR}/scripts/eval/eval_dsm.py" \
        --pred-dsm-path "${out}/test_opNone/ours_${ITERS}/dsm/${dsm}" \
        --gt-dir "${data}/truth/${SCENE}" --out-dir "${out}/" --aoi-id "${SCENE}" 2>/dev/null | grab_mae)
  mkdir -p "${out}/tree"
  mt=$(python "${EOGS_DIR}/scripts/eval/eval_dsm.py" \
        --pred-dsm-path "${out}/test_opNone/ours_${ITERS}/dsm/${dsm}" \
        --gt-dir "${data}/truth/${SCENE}" --out-dir "${out}/tree/" --aoi-id "${SCENE}" --filter_tree 2>/dev/null | grab_mae)
  echo "  -> w=${w}: overall MAE=${mo}  tree MAE=${mt}"
  echo "${SCENE},${w},${mo},${mt}" >> "${SUMMARY}"
done

echo
echo "==================== SWEEP DONE ===================="
column -t -s, "${SUMMARY}"
echo "saved: ${SUMMARY}"
