#!/usr/bin/env bash
# 04_prep_cameras.sh — generate affine approximations of the RPC camera models
# for each milestone scene, using EOGS's own dataset_creation script.
# Run after 03_get_eogs_data.sh, with the conda env active.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

SCENES="${1:-${MILESTONE_SCENES}}"   # pass a scene name to do just one, else all

if [ ! -d "${EOGS_DIR}/data/images" ]; then
  echo "!! No data at ${EOGS_DIR}/data. Run scripts/03_get_eogs_data.sh first."; exit 1
fi

cd "${EOGS_DIR}"
for scene in ${SCENES}; do
  echo "==================== to_affine: ${scene} ===================="
  python scripts/dataset_creation/to_affine.py --scene_name "${scene}"
done
echo "DONE. Cameras prepped for: ${SCENES}"
echo "Note: if you later hit KeyError 'centerofscene_ECEF', re-run this step."
