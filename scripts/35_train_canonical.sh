#!/usr/bin/env bash
# 35_train_canonical.sh — train ONLY the canonical EOGS++ config (rpc_ba + 3PAN) per scene, with the
# same expname reproduce_main uses (eogsplus_rpcba_<scene>_pan_3PAN) so scripts/31+32 find them.
# Use this instead of the full ablation matrix when you only need the per-view DSMs for uncertainty.
#   bash scripts/35_train_canonical.sh                       # default: the 5 missing scenes
#   bash scripts/35_train_canonical.sh "JAX_068 IARPA_003"   # custom list
set -eo pipefail
EOGS2_DIR="${EOGS2_DIR:-$HOME/eogs-src/EOGS2}"
SCENES="${1:-JAX_004 JAX_068 JAX_214 JAX_260 IARPA_003}"
export MPLBACKEND=Agg   # non-interactive matplotlib -> plt.show() will not open a window or block
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate eogsplus
cd "${EOGS2_DIR}/src/gaussiansplatting"
# belt-and-suspenders: neutralise the two blocking plt.show() calls (idempotent)
for pyf in eval/eval_dsm.py utils/plot_utils.py; do
  [ -f "$pyf" ] && sed -i -E 's/^([[:space:]]*)plt\.show\(\)/\1plt.close("all")  # disabled for unattended runs/' "$pyf"
done
for scene in ${SCENES}; do
  echo "==================== canonical EOGS++: ${scene} ===================="
  python full_eval_pan.py experiments=eogsplus.yaml mode=3PAN rpc_type=rpc_ba \
    scene=${scene} dataset=pan expname=eogsplus_rpcba_${scene}_pan_3PAN
done
echo "==================== DONE — canonical runs ready for scripts/31 + 32 ===================="
