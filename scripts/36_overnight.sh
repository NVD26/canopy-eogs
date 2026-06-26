#!/usr/bin/env bash
# 36_overnight.sh — unattended pipeline: train only the MISSING canonical EOGS++ runs, then run the
# full uncertainty analysis (U0 batch -> U1 calibration -> U2 selection precondition). Continues past
# any single failure; everything goes to stdout (redirect to a log when launching). Run via nohup.
set -o pipefail
EOGS2_DIR="${EOGS2_DIR:-$HOME/eogs-src/EOGS2}"
REPO="${REPO:-/mnt/c/Users/Navaneeth/Claude/Projects/PhD work}"
export MPLBACKEND=Agg
echo "############ OVERNIGHT RUN started $(date) ############"
echo "EOGS2=${EOGS2_DIR}  REPO=${REPO}"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate eogsplus
cd "${EOGS2_DIR}/src/gaussiansplatting"

echo "==== 0) make EOGS2 non-blocking (neutralise plt.show) + check deps ===="
for pyf in eval/eval_dsm.py utils/plot_utils.py; do
  [ -f "$pyf" ] && sed -i -E 's/^([[:space:]]*)plt\.show\(\)/\1plt.close("all")  # disabled/' "$pyf"
done
python -c "import scipy,rasterio,matplotlib" 2>/dev/null || pip install scipy rasterio matplotlib --quiet

ALL_SCENES="JAX_004 JAX_068 JAX_214 JAX_260 IARPA_001 IARPA_002 IARPA_003"
echo "==== 1) train MISSING canonical EOGS++ runs only ===="
for scene in ${ALL_SCENES}; do
  if [ -d "${EOGS2_DIR}/output/eogsplus_rpcba_${scene}_pan_3PAN/test_opNone" ]; then
    echo "  [skip] ${scene} canonical run already exists"
  else
    echo "-------------------- TRAIN ${scene} --------------------"
    python full_eval_pan.py experiments=eogsplus.yaml mode=3PAN rpc_type=rpc_ba \
      scene=${scene} dataset=pan expname=eogsplus_rpcba_${scene}_pan_3PAN \
      || echo "  !! ${scene} training FAILED (continuing)"
  fi
done

echo "==== 2) U0: per-scene uncertainty foundation (scripts/31) ===="
bash "${REPO}/scripts/31_uncertainty_batch.sh" || echo "  !! U0 batch FAILED (continuing)"

echo "==== 3) U1: leave-one-scene-out calibration (scripts/32) ===="
python "${REPO}/scripts/32_uncertainty_calibration.py" --auto --eogs2-dir "${EOGS2_DIR}" \
  || echo "  !! U1 calibration FAILED (continuing)"

echo "==== 4) U2 precondition: does image selection matter? (scripts/34, JAX_068, K=6, N=4) ===="
bash "${REPO}/scripts/34_selection_precondition.sh" JAX_068 6 4 \
  || echo "  !! U2 precondition FAILED (continuing)"

echo "############ OVERNIGHT RUN finished $(date) ############"
echo "READ TOMORROW: (2) per-scene rho/AUSE — rho>0 across scenes = foundation holds."
echo "               (3) COMBINED row beats single signals on AUSE + low ECE = calibrated uncertainty."
echo "               (4) MAE spread across random subsets — large spread = selection has leverage."
