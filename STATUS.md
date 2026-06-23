# STATUS — 3DGS for Earth Observation PhD (living lab notebook)

**Convention:** Read this FIRST at the start of every session. UPDATE it at the END
of every session, then `git commit` + `git push`. This file is the shared brain
between the laptop and the 4090 PC. Keep it honest — record failures, not just wins.

**Project:** Canopy-aware + spectral 3D Gaussian Splatting for Earth observation (UAH PhD).
**Active machine for this entry:** 4090 home PC.
**Last updated:** 2026-06-22.

---

## 0. Current focus (one sentence)

EOGS reproduced (DSM MAE matches paper). Next: read Paper-1 references and design the
GEDI/ICESat-2 canopy-aware extension — but no new code until advisor sign-off on the plan.

---

## 1. Milestone tracker

Legend: ☐ todo · ◐ in progress · ✅ done · ✗ blocked

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Repo scaffolding built (scripts, configs, STATUS workflow) | ✅ |
| M1 | Conda env `eogs` builds; EOGS cloned + CUDA kernels compiled | ✅ |
| M2 | EOGS release `data.zip` downloaded + extracted; cameras prepped | ✅ |
| M3 | `bash train.sh reproduceMain` runs to completion on the 4090 | ✅ (JAX scenes) |
| M4 | DSM MAE matches EOGS paper Table 1 (within reason) | ✅ mean ≈1.44 m vs paper ≈1.4 m |
| M5 | MAE + gotchas recorded here; committed + pushed | ◐ (recording now) |
| — | **Gate:** stop and report MAE before any new method work | ✅ reported |

---

## 2. Done so far

- 2026-06-22: Repo scaffolding created (Cowork session). Verified EOGS code repo
  (`github.com/mezzelfo/EOGS`) bundles DFC2019 tiles + DSM truth (`dataset_v01`).
- 2026-06-22: Full environment built on the 4090 via `scripts/01_setup_env.sh`
  (miniconda, conda ToS, apt build tools, torch cu121, EOGS, CUDA 12.1 toolkit,
  3DGS kernels). EOGS reproduced on the 4 JAX scenes; DSM MAE matches the paper.

## 3. In progress right now

- Nothing running. Pipeline validated. Awaiting decision on Paper-1 next step.

## 4. Next steps (ordered)

1. (optional) Full Table 1: `bash scripts/04_prep_cameras.sh` (now includes IARPA_001/002/003)
   then `bash scripts/05_run_eogs.sh reproduceMain` to also get the 3 IARPA scenes.
2. Read deeply: EOGS/EOGS++, ForestSplat, GEDI L2A/L2B + ICESat-2 ATL08 fusion refs.
3. Set up Earthdata Login: `python scripts/02_earthdata_auth.py` (free account first).
4. Design Paper-1 lidar-anchored height loss + two-surface (canopy/ground) decomposition
   as an ADDITION to the working EOGS pipeline. Get advisor sign-off before building.

## 5. Blockers / open questions

- None blocking. Note: build chain on a fresh WSL2 box needs, in order: build-essential,
  conda ToS accept, gcc-12 (CUDA 12.1 host), system CUDA 12.1 toolkit (not conda's 13.x).
  All now automated in `scripts/01_setup_env.sh`.

---

## 6. Results log

| Date | Scene | Method | DSM MAE (m) | Notes |
|------|-------|--------|-------------|-------|
| 2026-06-22 | JAX_004 | EOGS (our run) | 1.379 | reproduceMain, 5000 iters |
| 2026-06-22 | JAX_068 | EOGS (our run) | 1.093 | |
| 2026-06-22 | JAX_214 | EOGS (our run) | 1.734 | |
| 2026-06-22 | JAX_260 | EOGS (our run) | 1.554 | |
| 2026-06-22 | JAX mean | EOGS (our run) | **1.440** | matches paper's ~1.4 m |
| — | IARPA_001/002/003 | EOGS | pending | data present; re-run after camera prep |

EOGS paper Table 1 reference (read exact per-scene values from the PDF to fill):
JAX_004 ≈ __, JAX_068 ≈ __, JAX_214 ≈ __, JAX_260 ≈ __ ; reported avg ≈ 1.4 m.

---

## 7. Environment snapshot

- OS / shell: WSL2 Ubuntu on Windows (4090 PC), host `LYRA`
- GPU / driver: NVIDIA RTX 4090 (24 GB), driver 591.86 (nvidia-smi CUDA 13.1 capable)
- conda env: `eogs`, Python 3.10
- PyTorch: cu121 wheels (torch.version.cuda = 12.1); torch.cuda.is_available() = True
- Build toolchain: system CUDA toolkit 12.1 at `/usr/local/cuda-12.1`; host compiler gcc-12
  (note: a stray conda CUDA 13.3 exists in the env but is ignored — build uses 12.1 via CUDA_HOME)
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

- **2026-06-22** — EOGS reproduced on the 4090. Built full env via 01_setup_env.sh (fixed,
  in order: build-essential/make, conda ToS, nounset vs conda hooks, env-matched CUDA 12.1
  toolkit vs system 13.3, gcc-12 host compiler, CC-var clobber, IARPA camera prep). Ran
  reproduceMain: JAX DSM MAE = 1.379/1.093/1.734/1.554 (mean 1.440 m), matching the paper's
  ~1.4 m. All fixes folded into the scripts (single self-installing path for the advisor).
- **2026-06-22** — Cowork scaffolding session. Built git repo, scripts, configs, notebook.
- **2026-06-22** — Repo initialized; planning docs added. No code run yet.
