#!/usr/bin/env bash
# 34_selection_precondition.sh — [U2 precondition] does the CHOICE of training images matter?
# Trains EOGS++ on several RANDOM K-image subsets (via train.txt) and reports the DSM-MAE spread.
# Large spread => image selection has leverage (build the uncertainty-guided selector). Small spread
# => all views ~equally useful, selection is pointless. NO EOGS++ code change (only train.txt).
#
#   bash scripts/34_selection_precondition.sh JAX_068 6 4     # scene, K=subset size, N=#random subsets
# NOTE: first run is a TEST — verify the output-dir naming + MAE capture below match your EOGS2.
set -eo pipefail
EOGS2_DIR="${EOGS2_DIR:-$HOME/eogs-src/EOGS2}"
SCENE="${1:-JAX_068}"; K="${2:-6}"; N="${3:-4}"
MODE="${MODE:-3PAN}"; RPC="${RPC:-rpc_ba}"
SPLIT="${EOGS2_DIR}/data/train_test_split/${SCENE}"
[ -f "${SPLIT}/train.txt" ] || { echo "!! no ${SPLIT}/train.txt"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate eogsplus
cd "${EOGS2_DIR}/src/gaussiansplatting"

mapfile -t ALL < "${SPLIT}/train.txt"
echo "scene ${SCENE}: ${#ALL[@]} train images available; testing ${N} random subsets of size ${K}"
cp "${SPLIT}/train.txt" "${SPLIT}/train.txt.orig_bak"
restore(){ mv -f "${SPLIT}/train.txt.orig_bak" "${SPLIT}/train.txt" 2>/dev/null || true; }
trap restore EXIT

declare -a MAES
for n in $(seq 1 ${N}); do
  # random K-subset
  printf "%s\n" "${ALL[@]}" | shuf | head -n ${K} > "${SPLIT}/train.txt"
  echo "==================== subset ${n}/${N} (K=${K}) ===================="
  log="/tmp/sel_${SCENE}_${n}.log"
  python full_eval_pan.py experiments=eogsplus.yaml mode=${MODE} rpc_type=${RPC} scene=${SCENE} > "${log}" 2>&1 || { echo "  train failed; see ${log}"; continue; }
  mae=$(grep -oE "MAE: *[0-9.]+" "${log}" | head -1 | grep -oE "[0-9.]+")
  echo "  subset ${n}: DSM MAE = ${mae}   (images: $(tr '\n' ' ' < "${SPLIT}/train.txt"))"
  MAES+=("${mae}")
done
restore; trap - EXIT
echo "==================== RESULT ===================="
printf "MAEs: %s\n" "${MAES[*]}"
python3 - "${MAES[@]}" <<'PY'
import sys; v=[float(x) for x in sys.argv[1:] if x]
if len(v)>=2:
    import statistics as st
    print(f"  n={len(v)}  mean={st.mean(v):.3f}  std={st.pstdev(v):.3f}  range={max(v)-min(v):.3f}  [{min(v):.3f},{max(v):.3f}]")
    print("  => LARGE spread (range > ~0.2 m) means image selection MATTERS -> build the guided selector.")
    print("  => small spread means all views ~equally useful -> U2 weak, reconsider.")
PY
