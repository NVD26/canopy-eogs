#!/usr/bin/env bash
# 04_prep_cameras.sh — generate affine approximations of the RPC camera models
# for every scene that `train.sh reproduceMain` will train. reproduceMain trains
# 4 JAX + 3 IARPA scenes; ALL need affine_models.json or they fail at train time
# with "Could not recognize scene type at .../affine_models.json".
# Run after 03_get_eogs_data.sh, with the conda env active.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

# Default to the full reproduceMain set; pass a scene name to prep just one.
SCENES="${1:-${PREP_SCENES:-${MILESTONE_SCENES}}}"

if [ ! -d "${EOGS_DIR}/data/images" ]; then
  echo "!! No data at ${EOGS_DIR}/data. Run scripts/03_get_eogs_data.sh first."; exit 1
fi

cd "${EOGS_DIR}"
failed=""
for scene in ${SCENES}; do
  echo "==================== to_affine: ${scene} ===================="
  if ! python scripts/dataset_creation/to_affine.py --scene_name "${scene}"; then
    echo "   !! to_affine failed for ${scene} (scene may be absent from the data bundle)."
    failed="${failed} ${scene}"
  fi
done
echo "DONE. Attempted camera prep for: ${SCENES}"
[ -n "${failed}" ] && echo "Scenes that FAILED prep (will be skipped at train time):${failed}"
echo "Note: if you later hit KeyError 'centerofscene_ECEF', re-run this step."
