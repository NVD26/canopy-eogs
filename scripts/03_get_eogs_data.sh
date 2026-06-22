#!/usr/bin/env bash
# 03_get_eogs_data.sh — download the EOGS release data.zip (DFC2019 tiles + DSM
# truth) and extract it into the EOGS clone's data/ folder.
# This is the ONLY data needed for the reproduction milestone. ~big; downloads
# to the 4090 only (gitignored, tracked in data/manifest.csv).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../configs/milestone.env"

if [ ! -d "${EOGS_DIR}" ]; then
  echo "!! EOGS not cloned yet at ${EOGS_DIR}. Run scripts/01_setup_env.sh first."; exit 1
fi

DATA_DIR="${EOGS_DIR}/data"
mkdir -p "${DATA_DIR}"
ZIP_PATH="${DATA_DIR}/data.zip"

if [ -d "${DATA_DIR}/images" ] && [ -d "${DATA_DIR}/truth" ]; then
  echo "Data already extracted at ${DATA_DIR} (images/ and truth/ present) — skipping download."
  exit 0
fi

echo "Downloading EOGS dataset_v01 -> ${ZIP_PATH}"
if command -v wget >/dev/null 2>&1; then
  wget -c -O "${ZIP_PATH}" "${EOGS_DATA_ZIP_URL}"
else
  curl -L -C - -o "${ZIP_PATH}" "${EOGS_DATA_ZIP_URL}"
fi

echo "Extracting into ${DATA_DIR} ..."
unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}"
rm -f "${ZIP_PATH}"

echo "Expected structure: ${DATA_DIR}/{images,rpcs,truth}/JAX_004 ..."
ls -la "${DATA_DIR}"
echo "DONE. Now run scripts/04_prep_cameras.sh"
