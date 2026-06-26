#!/usr/bin/env bash
# 27_erank_ablation.sh — [2DGS PRECONDITION, free/no-CUDA] does a SURFEL (flat-disk) prior help
# DSM accuracy on sparse oblique satellite views? EOGS's L_erank pushes Gaussians toward 2D disks
# (default w_L_erank=0). Sweep it; compare VEG/BLDG DSM MAE to the erank=0 baseline.
# Necessary-not-sufficient: if even flattening helps, full 2DGS is promising; if it HURTS, the
# sparse oblique regime may dislike flat primitives -> reconsider BEFORE building a CUDA rasterizer.
#   bash scripts/27_erank_ablation.sh JAX_214
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "${HERE}/.." && pwd)"
EOGS_DIR="${EOGS_DIR:-$HOME/eogs-src/EOGS}"; SCENE="${1:-JAX_214}"; ITERS="${NUMITER:-5000}"
WEIGHTS="${WEIGHTS:-0.1 0.5}"   # erank=0.0 baseline = the standard EOGS run (compare against it)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate eogs
cd "${EOGS_DIR}/src/gaussiansplatting"; D="${EOGS_DIR}/data"; ts=$(date +%s)
echo "==== 2DGS precondition on ${SCENE}: w_L_erank in {0.0(baseline) ${WEIGHTS}} ===="
echo "  (erank=0.0 baseline for JAX_214 from the #5 run was: VEG 1.586 | BLDG 2.796 | overall 1.722)"
for w in ${WEIGHTS}; do
  exp="erank_${ts}_${SCENE}_w${w}"; out="${EOGS_DIR}/output/${exp}"
  echo "-------------------- TRAIN w_L_erank=${w} --------------------"
  python train.py -s "${D}/affine_models/${SCENE}" --images "${D}/images/${SCENE}" --eval \
    -m "${out}" --sh_degree 0 --iterations "${ITERS}" --w_L_erank "${w}"
  python render.py -m "${out}"
  dsm=$(ls "${out}/test_opNone/ours_${ITERS}/dsm/" | sort -V | tail -1)
  python "${EOGS_DIR}/scripts/eval/eval_dsm.py" \
    --pred-dsm-path "${out}/test_opNone/ours_${ITERS}/dsm/${dsm}" \
    --gt-dir "${D}/truth/${SCENE}" --out-dir "${out}/" --aoi-id "${SCENE}" >/dev/null 2>&1 || true
  python "${REPO}/scripts/25_tree_vs_building_eval.py" --rdsm "${out}/${SCENE}_rdsm.tif" \
    --scene "${SCENE}" --eogs-dir "${EOGS_DIR}" --tag "w_L_erank=${w}"
done
echo "==== READ: if a surfel prior LOWERS DSM MAE vs the 1.722 baseline -> 2DGS promising. ===="
echo "====       if it RAISES it -> early FAIL signal; reconsider before any CUDA rasterizer. ===="
