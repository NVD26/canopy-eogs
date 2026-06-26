#!/usr/bin/env bash
# 31_uncertainty_batch.sh — run U0 (scripts/30) across ALL reproduced EOGS++ scenes and collect a
# per-scene Spearman/AUSE table. Auto-detects scene name + the (early-stopped) iteration count from
# each output dir. Run after experiments/reproduce_main.sh finishes.
#   bash scripts/31_uncertainty_batch.sh
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "${HERE}/.." && pwd)"
EOGS2_DIR="${EOGS2_DIR:-$HOME/eogs-src/EOGS2}"
PAT="${1:-eogsplus_rpcba_*_pan_3PAN}"     # canonical EOGS++ per scene (rpc_ba + 3PAN)     # which output dirs to analyse (default the EOGS++ runs)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate eogsplus
echo "==== U0 across scenes (pattern: ${PAT}) ===="
shopt -s nullglob
for exp in "${EOGS2_DIR}"/output/${PAT}; do
  [ -d "${exp}/test_opNone" ] || continue
  scene=$(basename "${exp}" | grep -oE 'JAX_[0-9]+|IARPA_[0-9]+' | head -1)
  oursdir=$(ls -d "${exp}/test_opNone/ours_"* 2>/dev/null | sort -V | tail -1)
  [ -n "${scene}" ] && [ -n "${oursdir}" ] || { echo "  skip $(basename "${exp}") (no scene/ours dir)"; continue; }
  iters=$(basename "${oursdir}" | sed 's/ours_//')
  echo "==================== ${scene}  (iters ${iters}) ===================="
  python "${REPO}/scripts/30_uncertainty_precondition.py" \
    --exp "${exp}" --scene "${scene}" --iters "${iters}" --eogs2-dir "${EOGS2_DIR}" 2>&1 \
    | grep -E "per-view DSMs|MAE|disagreement|roughness|VEGETATION|BUILDING|n per-view" || echo "  (scene failed — check manually)"
done
echo "==== DONE. Collect the rho/AUSE per scene; consistent rho>0 across scenes = solid foundation. ===="
