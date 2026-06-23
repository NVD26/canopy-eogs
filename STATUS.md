# STATUS — 3DGS for Earth Observation PhD (living lab notebook)

**Convention:** Read this FIRST at the start of every session. UPDATE it at the END
of every session, then `git commit` + `git push`. This file is the shared brain
between the laptop and the 4090 PC. Keep it honest — record failures, not just wins.

**Project:** Canopy-aware + spectral 3D Gaussian Splatting for Earth observation (UAH PhD).
**Active machine for this entry:** 4090 home PC.
**Last updated:** 2026-06-22.

---

## 0. Current focus (one sentence)

EOGS fully reproduced (all 7 DFC2019/IARPA scenes, DSM MAE matches paper). Paused before
Paper-1 (GEDI/ICESat-2 canopy-aware extension) — design + advisor sign-off next session.

---

## 1. Milestone tracker

Legend: ☐ todo · ◐ in progress · ✅ done · ✗ blocked

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Repo scaffolding built (scripts, configs, STATUS workflow) | ✅ |
| M1 | Conda env `eogs` builds; EOGS cloned + CUDA kernels compiled | ✅ |
| M2 | EOGS release `data.zip` downloaded + extracted; cameras prepped (all 7) | ✅ |
| M3 | `train.sh reproduceMain` runs to completion (all 7 scenes) | ✅ |
| M4 | DSM MAE matches EOGS paper Table 1 (within reason) | ✅ JAX mean 1.44 m vs paper ~1.4 m |
| M5 | MAE + gotchas recorded here; committed + pushed | ✅ |
| — | **Gate:** stop and report MAE before any new method work | ✅ passed |

**PHASE-1 MILESTONE COMPLETE — validated EOGS baseline on the 4090.**

---

## 2. Done so far

- 2026-06-22: Repo scaffolding created. Verified EOGS code repo bundles DFC2019 +
  IARPA tiles + DSM truth (`dataset_v01`).
- 2026-06-22: Full environment built on the 4090 via `scripts/01_setup_env.sh`
  (miniconda, conda ToS, apt build tools, torch cu121, EOGS, CUDA 12.1 toolkit, kernels).
- 2026-06-22: EOGS reproduced on all 7 scenes (4 JAX + 3 IARPA); DSM MAE matches paper.

## 3. In progress right now

- Nothing running. Phase-1 done. Paper-1 design is the next work item (next session).

## 4. Next steps (ordered)

1. Read deeply: EOGS/EOGS++, ForestSplat, GEDI L2A/L2B + ICESat-2 ATL08 fusion refs.
2. Earthdata Login set up (`scripts/02_earthdata_auth.py`) — for GEDI/ICESat-2/HLS.
3. Select forested AOIs with GEDI + ICESat-2 + USGS 3DEP airborne-lidar overlap.
4. Design Paper-1: lidar-anchored height loss + two-surface (canopy/ground) decomposition
   + calibrated uncertainty head, as an ADDITION to EOGS. Advisor sign-off before building.

## 5. Blockers / open questions

- None. Build chain on a fresh WSL2 box (all automated in `01_setup_env.sh`, in order):
  build-essential/make, conda ToS accept, drop `set -u` (conda hooks), env-matched system
  CUDA 12.1 toolkit (ignore conda's 13.3), gcc-12 host compiler, camera prep for all 7 scenes.

---

## 6. Results log — EOGS reproduction (Table 1)

Full run `reproduceMain` (log eogs_reproduceMain_20260622_194045), 5000 iters/scene:

| Date | Scene | Method | DSM MAE (m) |
|------|-------|--------|-------------|
| 2026-06-22 | JAX_004   | EOGS (our run) | 1.360 |
| 2026-06-22 | JAX_068   | EOGS (our run) | 1.095 |
| 2026-06-22 | JAX_214   | EOGS (our run) | 1.784 |
| 2026-06-22 | JAX_260   | EOGS (our run) | 1.539 |
| 2026-06-22 | IARPA_001 | EOGS (our run) | 1.591 |
| 2026-06-22 | IARPA_002 | EOGS (our run) | 1.985 |
| 2026-06-22 | IARPA_003 | EOGS (our run) | 2.072 |
| 2026-06-22 | **JAX mean**   | EOGS (our run) | **1.444** |
| 2026-06-22 | **IARPA mean** | EOGS (our run) | **1.883** |
| 2026-06-22 | **All-7 mean** | EOGS (our run) | **1.632** |

Note: a first JAX-only run (log _191754) gave 1.379/1.093/1.734/1.554 (mean 1.440) — the
small per-scene differences are the random-seed variation the EOGS README flags. Stable.
EOGS paper Table 1 reference (fill exact per-scene from the PDF): JAX avg ≈ 1.4 m.

---

## 7. Environment snapshot

- OS / shell: WSL2 Ubuntu on Windows (4090 PC), host `LYRA`
- GPU / driver: NVIDIA RTX 4090 (24 GB), driver 591.86 (nvidia-smi CUDA 13.1 capable)
- conda env: `eogs`, Python 3.10
- PyTorch: cu121 wheels (torch.version.cuda = 12.1); torch.cuda.is_available() = True
- Build toolchain: system CUDA toolkit 12.1 at `/usr/local/cuda-12.1`; host compiler gcc-12
  (a stray conda CUDA 13.3 exists in the env but is ignored — build uses 12.1 via CUDA_HOME)
- EOGS repo commit: cca973e7ea512091b52c8ff741c80ddade5793d2
- diff-gaussian-rasterization / simple-knn: built ✅
- CUDA wheel index used: cu121

## 8. Data inventory (what's downloaded where — never synced to laptop)

| Dataset | Location on 4090 | Source | Pulled? |
|---------|------------------|--------|---------|
| EOGS dataset_v01 (JAX_004/068/214/260 + IARPA_001/002/003, images+rpcs+truth DSM) | `~/eogs-src/EOGS/data` | EOGS GitHub release | ✅ |
| GEDI L2A/L2B (Paper 1, later) | — | Earthdata / earthaccess | ☐ |
| ICESat-2 ATL08 (Paper 1, later) | — | Earthdata / earthaccess | ☐ |
| HLS (Paper 2, later) | — | Earthdata / earthaccess | ☐ |

---

## 9. Session log (newest on top)

- **2026-06-22** — Full EOGS reproduction on the 4090: all 7 scenes via reproduceMain.
  JAX MAE mean 1.44 m (matches paper ~1.4 m); IARPA mean 1.88 m. Env built end-to-end by
  01_setup_env.sh; every setup error folded back into the scripts (single self-installing
  path for the advisor). Phase-1 milestone complete. Earthdata auth: pending (next session).
- **2026-06-22** — Cowork scaffolding session. Built git repo, scripts, configs, notebook.
- **2026-06-22** — Repo initialized; planning docs added. No code run yet.
