#!/usr/bin/env bash
# run_milestone.sh — end-to-end milestone driver: data -> cameras -> reproduce.
# Assumes 01_setup_env.sh already built the env + cloned/compiled EOGS.
# Activate the env first:  conda activate eogs
#
#   bash scripts/run_milestone.sh            # all milestone scenes, reproduceMain
#   bash scripts/run_milestone.sh smoke      # just JAX_004 camera prep + a single run
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

MODE="${1:-full}"

echo "########## STEP 1/3: data ##########"
bash "${HERE}/03_get_eogs_data.sh"

echo "########## STEP 2/3: camera prep ##########"
if [ "${MODE}" = "smoke" ]; then
  bash "${HERE}/04_prep_cameras.sh" "${SMOKE_SCENE}"
else
  bash "${HERE}/04_prep_cameras.sh"
fi

echo "########## STEP 3/3: reproduce EOGS ##########"
# reproduceMain reproduces the paper's Table 1 across the bundled scenes.
bash "${HERE}/05_run_eogs.sh" reproduceMain

echo
echo "########## MILESTONE RUN COMPLETE ##########"
echo "1. Read the MAE from the run log in results/logs/ (grep -i mae)."
echo "2. Record MAE + train time in STATUS.md §6; fill §7 env snapshot + §8 data inventory."
echo "3. git add -A && git commit -m 'EOGS reproduction: MAE=<...>' && git push"
echo "4. STOP and report the MAE before building anything new."
