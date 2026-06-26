#!/usr/bin/env bash
# 26_opacity_ablation.sh — [Idea-5 PRECONDITION] does EOGS's opacity penalty trade buildings vs
# vegetation? Trains a scene at several w_L_opacity, evals VEG vs BUILDING DSM MAE. NO code change.
# Idea #5 survives ONLY if lowering w_L_opacity IMPROVES veg MAE while DEGRADING building MAE.
#   bash scripts/26_opacity_ablation.sh JAX_214
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "${HERE}/.." && pwd)"
EOGS_DIR="${EOGS_DIR:-$HOME/eogs-src/EOGS}"; SCENE="${1:-JAX_214}"; ITERS="${NUMITER:-5000}"
WEIGHTS="${WEIGHTS:-0.10 0.0}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate eogs
cd "${EOGS_DIR}/src/gaussiansplatting"; D="${EOGS_DIR}/data"; ts=$(date +%s)
echo "==== Idea-5 precondition on ${SCENE}: w_L_opacity in {${WEIGHTS}} ===="
for w in ${WEIGHTS}; do
  exp="op_abl_${ts}_${SCENE}_w${w}"; out="${EOGS_DIR}/output/${exp}"
  echo "-------------------- TRAIN w_L_opacity=${w} --------------------"
  python train.py -s "${D}/affine_models/${SCENE}" --images "${D}/images/${SCENE}" --eval \
    -m "${out}" --sh_degree 0 --iterations "${ITERS}" --w_L_opacity "${w}"
  python render.py -m "${out}"
  dsm=$(ls "${out}/test_opNone/ours_${ITERS}/dsm/" | sort -V | tail -1)
  python "${EOGS_DIR}/scripts/eval/eval_dsm.py" \
    --pred-dsm-path "${out}/test_opNone/ours_${ITERS}/dsm/${dsm}" \
    --gt-dir "${D}/truth/${SCENE}" --out-dir "${out}/" --aoi-id "${SCENE}" >/dev/null 2>&1 || true
  python "${REPO}/scripts/25_tree_vs_building_eval.py" --rdsm "${out}/${SCENE}_rdsm.tif" \
    --scene "${SCENE}" --eogs-dir "${EOGS_DIR}" --tag "w_L_opacity=${w}"
done
echo "==== READ: idea #5 lives only if lower w improves VEG MAE AND worsens BLDG MAE (a real tradeoff). ===="
