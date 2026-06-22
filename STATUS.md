# STATUS — 3DGS for Earth Observation PhD (living lab notebook)

**Convention:** Read this FIRST at the start of every session. UPDATE it at the END
of every session, then `git commit` + `git push`. This file is the shared brain
between the laptop and the 4090 PC. Keep it honest — record failures, not just wins.

**Project:** Canopy-aware + spectral 3D Gaussian Splatting for Earth observation (UAH PhD).
**Active machine for this entry:** 4090 home PC.
**Last updated:** 2026-06-22.

---

## 0. Current focus (one sentence)

Reproduce EOGS on DFC2019 tiles and match its DSM MAE. Nothing new until this passes.

---

## 1. Milestone tracker

Legend: ☐ todo · ◐ in progress · ✅ done · ✗ blocked

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Repo scaffolding built (scripts, configs, STATUS workflow) | ✅ |
| M1 | Conda env `eogs` builds; EOGS cloned + CUDA kernels compiled | ☐ |
| M2 | EOGS release `data.zip` downloaded + extracted; cameras prepped | ☐ |
| M3 | `bash train.sh reproduceMain` runs to completion on the 4090 | ☐ |
| M4 | DSM MAE matches EOGS paper Table 1 (within reason) | ☐ |
| M5 | MAE + gotchas recorded here; committed + pushed | ☐ |
| — | **Gate:** stop and report MAE before any new method work | ☐ |

---

## 2. Done so far

- 2026-06-22: Repo scaffolding created (Cowork session). Verified the EOGS code repo
  (`github.com/mezzelfo/EOGS`) bundles the DFC2019 milestone tiles + DSM truth in its
  `dataset_v01` release, and reproduces via `bash train.sh reproduceMain`. No IEEE
  DataPort / Earthdata needed for the milestone. Wrote setup/data/run/eval scripts.

## 3. In progress right now

- (nothing yet — run `scripts/00_check_gpu.sh` then `scripts/01_setup_env.sh` on the 4090)

## 4. Next steps (ordered)

1. `bash scripts/init_git.sh` — clean repo on `main`, first commit (+ optional GitHub remote).
2. `bash scripts/00_check_gpu.sh` — confirm GPU + driver + (after env) torch.cuda.
3. `bash scripts/01_setup_env.sh` — build conda env `eogs`, clone EOGS, compile CUDA kernels.
4. `bash scripts/run_milestone.sh` — download data, prep cameras, `train.sh reproduceMain`.
5. Read the MAE EOGS prints; optionally cross-check with `scripts/06_eval_dsm_mae.py`.
6. Record MAE + gotchas in §6 and §9 below; fill §7 env snapshot + §8 data inventory.
7. Commit + push. **Stop and report the MAE.**

## 5. Blockers / open questions

- (none yet) — likely first friction points: CUDA toolkit / `nvcc` for building
  `diff-gaussian-rasterization` on WSL2; torch CUDA wheel matching the 4090 driver.

---

## 6. Results log

| Date | Scene(s) | Method | DSM MAE (m) | Train time | Notes |
|------|----------|--------|-------------|-----------|-------|
| — | — | EOGS (paper, target) | _fill from Table 1_ | — | reference to beat/match |
| — | — | EOGS (our run) | _TBD_ | _TBD_ | first reproduction |

EOGS paper Table 1 reference MAE (fill exact values when you read the PDF):
JAX_004 ≈ __, JAX_068 ≈ __, JAX_214 ≈ __, JAX_260 ≈ __.

---

## 7. Environment snapshot (fill after setup)

- OS / shell: WSL2 Ubuntu ____ on Windows (4090 PC)
- GPU / driver / CUDA: ____ (from `nvidia-smi`)
- conda env: `eogs`, Python 3.10
- Key versions: torch ____, torchvision ____, rasterio ____, rpcm ____, earthaccess ____
- EOGS repo commit: ____  (record `git -C ~/eogs-src/EOGS rev-parse HEAD`)
- diff-gaussian-rasterization / simple-knn: built ☐
- CUDA wheel index used: ____ (cu121 / cu124)

## 8. Data inventory (what's downloaded where — never synced to laptop)

| Dataset | Location on 4090 | Size | Source | Pulled? |
|---------|------------------|------|--------|---------|
| EOGS dataset_v01 (JAX/IARPA tiles + DSM truth) | `~/eogs-src/EOGS/data` | ~ | EOGS GitHub release | ☐ |
| GEDI L2A/L2B (Paper 1, later) | — | — | Earthdata / earthaccess | ☐ |
| ICESat-2 ATL08 (Paper 1, later) | — | — | Earthdata / earthaccess | ☐ |
| HLS (Paper 2, later) | — | — | Earthdata / earthaccess | ☐ |

---

## 9. Session log (newest on top)

- **2026-06-22** — Cowork scaffolding session. Built git repo, scripts, configs, and this
  notebook. Confirmed milestone path uses the EOGS release-bundled data (no extra accounts).
  Next: run `00_check_gpu.sh` → `01_setup_env.sh` on the 4090.
- **2026-06-22** — Repo initialized; planning docs added. No code run yet.
