# data/ — policy

**Big data is NEVER committed and NEVER synced laptop<->4090.** Only this README
and `manifest.csv` live in git. Everything else here is gitignored.

- The EOGS reproduction data (DFC2019 JAX/IARPA tiles + DSM ground truth) is
  downloaded by `scripts/03_get_eogs_data.sh` into the EOGS clone at
  `${EOGS_DIR}/data` (default `~/eogs-src/EOGS/data`), **not** here.
- Paper 1/2 data (GEDI L2A/L2B, ICESat-2 ATL08, HLS) will be pulled with
  `earthaccess` to a local cache on the 4090. Record each pull in `manifest.csv`.

When you download anything, add a row to `manifest.csv` so the other machine
(and future you) knows what exists and where — without ever moving the bytes.
