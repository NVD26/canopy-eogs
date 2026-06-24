#!/usr/bin/env bash
# 15_train_lidar.sh — [Paper 1 / M7] train EOGS WITH the GEDI canopy-anchor loss and
# compare against the unsupervised baseline on the SAME scene & seed. First applies the
# loss patch to the EOGS clone (idempotent), then for each condition runs
# train -> render -> eval DSM MAE (overall AND tree-pixels-only via --filter_tree).
#
#   bash scripts/15_train_lidar.sh [SCENE] [W_LIDAR]
# defaults: SCENE=JAX_068  W_LIDAR=0.1   (override anchors with ANCHORS=..., iters with NUMITER=...)
#
# NOTE: `set -eo pipefail` WITHOUT `-u` — conda activate hooks reference unbound vars.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
EOGS_DIR="${EOGS_DIR:-$HOME/eogs-src/EOGS}"
SCENE="${1:-JAX_068}"
W_LIDAR="${2:-0.1}"
ITERS="${NUMITER:-5000}"
ANCHORS="${ANCHORS:-${REPO_ROOT}/data/anchors/${SCENE}_gedi_anchors.npz}"

[ -d "${EOGS_DIR}/data/affine_models/${SCENE}" ] || {
  echo "!! ${SCENE} not prepped at ${EOGS_DIR}/data/affine_models/${SCENE}. Run scripts/04_prep_cameras.sh ${SCENE}"; exit 1; }
[ -f "${ANCHORS}" ] || {
  echo "!! anchors not found: ${ANCHORS}"; echo "   build them: python scripts/12_build_anchors.py --scene ${SCENE} --aoi-km 2"; exit 1; }

echo "==================== applying Paper-1 lidar-loss patch ===================="
python "${REPO_ROOT}/eogs_mods/apply_paper1_lidar.py" "${EOGS_DIR}"

cd "${EOGS_DIR}/src/gaussiansplatting"   # EOGS runs train.py/render.py from here
data="${EOGS_DIR}/data"
ts=$(date +%s)

run_one () {   # $1 = tag (baseline|lidar)   $2 = w_L_lidar
  local tag="$1" w="$2"
  local exp="p1lidar_${ts}_${SCENE}_${tag}"
  local out="${EOGS_DIR}/output/${exp}"
  echo "==================== TRAIN ${tag}  (w_L_lidar=${w}) ===================="
  python train.py \
    -s "${data}/affine_models/${SCENE}" \
    --images "${data}/images/${SCENE}" \
    --eval -m "${out}" \
    --sh_degree 0 --iterations "${ITERS}" \
    --lidar_anchors_path "${ANCHORS}" \
    --w_L_lidar "${w}" --iterstart_L_lidar 0
  python render.py -m "${out}"
  local dsm
  dsm=$(ls "${out}/test_opNone/ours_${ITERS}/dsm/" | sort -V | tail -n1)
  echo "-------------------- EVAL ${tag}: OVERALL --------------------"
  python "${EOGS_DIR}/scripts/eval/eval_dsm.py" \
    --pred-dsm-path "${out}/test_opNone/ours_${ITERS}/dsm/${dsm}" \
    --gt-dir "${data}/truth/${SCENE}" --out-dir "${out}/" --aoi-id "${SCENE}"
  echo "-------------------- EVAL ${tag}: TREE PIXELS (--filter_tree) --------------------"
  mkdir -p "${out}/tree"
  python "${EOGS_DIR}/scripts/eval/eval_dsm.py" \
    --pred-dsm-path "${out}/test_opNone/ours_${ITERS}/dsm/${dsm}" \
    --gt-dir "${data}/truth/${SCENE}" --out-dir "${out}/tree/" --aoi-id "${SCENE}" --filter_tree
  echo "RESULT ${tag}: ${out}"
}

run_one baseline 0.0
run_one lidar    "${W_LIDAR}"

echo
echo "==================== DONE — ablation complete ===================="
echo "Compare the two EVAL prints (overall + tree) above:"
echo "  baseline = EOGS unchanged (w_L_lidar=0);  lidar = + GEDI canopy-anchor loss (w=${W_LIDAR})."
echo "Same scene, same code path, only w_L_lidar differs. Record both numbers in STATUS.md."
