#!/usr/bin/env bash
# 05_run_eogs.sh — run the EOGS reproduction. This is the milestone.
#   default:        bash scripts/05_run_eogs.sh                 -> train.sh reproduceMain
#   custom target:  bash scripts/05_run_eogs.sh <train.sh-arg>  -> train.sh <arg>
#
# `reproduceMain` reproduces Table 1 of the paper (the MAE numbers to match).
# EOGS prints/saves DSM MAE per scene. Logs are tee'd into results/logs/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

TARGET="${1:-reproduceMain}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOG_DIR}/eogs_${TARGET}_${STAMP}.log"

if [ ! -f "${EOGS_DIR}/train.sh" ]; then
  echo "!! ${EOGS_DIR}/train.sh not found. Did setup (01) succeed?"; exit 1
fi

echo "Running EOGS '${TARGET}' in ${EOGS_DIR}"
echo "Logging to ${LOGFILE}"
cd "${EOGS_DIR}"
# Record the exact commit alongside the log for reproducibility.
echo "EOGS commit: $(git rev-parse HEAD)" | tee "${LOGFILE}"
set -o pipefail
bash train.sh "${TARGET}" 2>&1 | tee -a "${LOGFILE}"

echo
echo "DONE. Grep the log for the MAE numbers and copy them into STATUS.md §6:"
echo "   grep -iE 'mae|altitude|error' '${LOGFILE}'"
echo "If EOGS wrote predicted DSM .tif files, you can also cross-check with:"
echo "   python scripts/06_eval_dsm_mae.py --pred <pred_dsm.tif> --truth <truth_dsm.tif>"
