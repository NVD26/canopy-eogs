# STATUS — 3DGS for Earth Observation PhD (living lab notebook)

**Convention:** Read this FIRST at the start of every session. UPDATE it at the END
of every session, then `git commit` + `git push`. This file is the shared brain
between the laptop and the 4090 PC. Keep it honest — record failures, not just wins.

**Project:** Canopy-aware + spectral 3D Gaussian Splatting for Earth observation (UAH PhD).
**Active machine for this entry:** 4090 home PC.
**Last updated:** 2026-06-22.

---

## 0. Current focus (one sentence)

Paper-1 (canopy-aware EOGS): design done, two-surface idea validated (prototypes), and GEDI GROUND
VALIDATED on real data vs 3DEP (datum -29.6 m=geoid, 0.54 m median, 90% within 3 m). Next: fold
datum+filter into the anchor builder (wide AOI), then M7 = lidar loss in EOGS.

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
2. ~~Earthdata Login~~ ✅ done (auth verified, ~/.netrc persisted) — ready for GEDI/ICESat-2/HLS pulls.
3. Select forested AOIs with GEDI + ICESat-2 + USGS 3DEP airborne-lidar overlap.
4. Design Paper-1: lidar-anchored height loss + two-surface (canopy/ground) decomposition
   + calibrated uncertainty head, as an ADDITION to EOGS. Advisor sign-off before building.

## 5. Blockers / open questions

- **Lidar density (CORRECTED):** the 108 GEDI count used a ~800 m padded bbox, but the ACTUAL
  eval tile is only **256 m** (JAX_068_DSM.txt = 512 px @ 0.5 m, UTM zone 17). True in-tile anchor
  count is being recomputed by scripts/12_build_anchors.py against the exact DSM bounds — expect
  meaningfully fewer than 108 (~tens). Still likely workable (prototype: ~20-70 anchors), but if a
  256 m tile is too anchor-poor we enlarge the AOI. GEDI primary; ICESat-2 opportunistic (0 in
  JAX_068). LESSON: always use exact tile bounds, not padded bbox, for the budget.
- **Validation = INCONCLUSIVE (honest):** scripts/12_build_anchors.py found only **7 quality GEDI
  footprints** in the 256 m JAX_068 tile (quality_flag limited, not sensitivity). The canopy-top vs
  airborne-DSM-max check gave ~0 m offset but MAE 5.7 m / std 7.5 m — too few points + too much
  scatter to TRUST. Gate now reports INCONCLUSIVE (needs >=30 footprints) and marks anchors
  UNVALIDATED. Two real implications: (a) a 256 m tile is anchor-POOR for supervision AND for
  validation; (b) the GEDI<->DSM vertical datum is not yet pinned.
- **GEDI GROUND VALIDATED (3DEP, 10 km AOI, scripts/13):** 17,182 GEDI footprints vs USGS 3DEP
  bare-earth DTM. Datum offset = **-29.61 m** (= local geoid, exactly as expected) -> projection +
  datum are CORRECT. Residual after offset: **abs-median 0.54 m** (sub-meter agreement!). CAVEAT:
  mean MAE 68 m / std 569 m -> a minority of GEDI shots are GROSS BLUNDERS (known GEDI behavior,
  not removed by quality_flag alone). RESULT after datum shift: **90.0% within 3 m, 72.8% within 1 m
  -> 15,462 / 17,182 CLEAN anchors**; MAD 0.80 m. So GEDI ground = sub-meter ground truth after
  (1) +(-29.61 m) datum shift and (2) drop |resid|>3 m blunders (~10%). VERDICT: PASS. 3DEP DTM also
  becomes our bare-earth ground-truth for the under-canopy evaluation. Saved gedi_3dep_validation.npz.
- **Anchor pipeline TODO:** fold the -29.61 m datum shift + outlier filter into scripts/12; the
  big-AOI 3DEP also resolves the 256 m-tile anchor-poverty (thousands of footprints over Jax).
- **EOGS tree masks:** ship per-scene (scripts/eval/tree_masks/<scene>.png); JAX_068 mask is ~96%
  of pixels (polarity to confirm) -> JAX is heavily vegetated, good canopy site. Use for measuring
  EOGS error on tree pixels (motivation) and as a learned-mask prior.
- **AOI finding:** RPC localization shows JAX tiles = Jacksonville FL (3DEP airborne lidar
  available for dense CHM/DTM truth); IARPA tiles = Buenos Aires, Argentina (NO USGS 3DEP).
  => center canopy evaluation on JAX; treat IARPA as built-up no-regression checks.

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
| GEDI L2A/L2B (Paper 1) | ~/eogs-data/lidar_probe | Earthdata / earthaccess | **108 footprints/JAX_068 tile** (full 24-granule aggregate); primary anchor source |
| ICESat-2 ATL08 (Paper 1) | (probe) | Earthdata / earthaccess | sparse at tile scale: 0 in JAX_068, 13 in JAX_004 (km track spacing); opportunistic |
| HLS (Paper 2, later) | — | Earthdata / earthaccess | ☐ |

---

## 9. Session log (newest on top)

- **2026-06-22** — M6 DATA VALIDATED. scripts/13_validate_gedi_3dep.py: fetched USGS 3DEP bare-earth
  DTM over a 10 km Jacksonville AOI, validated 17,182 GEDI ground returns. Datum offset -29.61 m
  (= local geoid, confirms projection+datum correct); residual abs-median 0.54 m, MAD 0.80 m;
  90.0% within 3 m, 72.8% within 1 m -> 15,462 clean anchors. Caught (via mean MAE 68 m vs median
  0.54 m) that ~10% of GEDI shots are gross blunders -> must datum-shift AND outlier-filter. The
  256 m DFC2019 tile had only 7 GEDI footprints (anchor-poor); the wide 3DEP AOI fixes both the
  validation power and the supervision budget. GEDI ground is now TRUSTED ground truth. Next:
  fold -29.61 m + filter into scripts/12 (wide AOI); then M7 lidar loss.

- **2026-06-22** — Paper-1 prototype v2 (prototypes/twosurface_v2_learned_opacity.py): removed
  the known-canopy-mask assumption. Mask now LEARNED jointly from a noisy NDVI-like cue + sparse
  lidar two-returns + smoothness (IoU 0.94 vs true forest, beating the 0.90 cue alone). Under-canopy
  DTM MAE: single-surface 12.18 m, oracle-mask 0.198 m, LEARNED-mask 0.223 m. Learned ≈ oracle ≫
  single -> method needs no given canopy mask. Last conceptual gap before EOGS integration closed.
  Also: ran scripts/10_query_lidar.py --count-footprints (2 granules): GEDI 0-34 footprints/tile,
  ATL08 0-13/tile per 2 overpasses (highly variable); full JAX_068 aggregation (25 granules) running.

- **2026-06-22** — Paper-1 kickoff. Novelty fact-check (niche still open: spaceborne-lidar +
  satellite-RPC 3DGS canopy/ground separation is unoccupied). Mapped EOGS injection points
  (renders altitude channel; has disabled L_altitude_reference + L_nll uncertainty hooks).
  Wrote advisor design doc (Paper1_Design_CanopyAwareEOGS.docx), lidar query script
  (scripts/10_query_lidar.py), and a CPU feasibility prototype (prototypes/twosurface_toy.py):
  with the SAME 4% sparse lidar anchors, two-surface model recovers under-canopy DTM to 0.20 m
  MAE while single-surface (EOGS-like) is off by ~12 m (the canopy height); canopy-top fit equal
  (no regression). Density sweep monotonic (0%→1.10 m, 1%→0.48, 4%→0.21, 8%→0.10). Core idea
  is identifiable from sparse anchors — green light to design the real method. Gate: advisor
  sign-off before EOGS integration.

- **2026-06-22** — Full EOGS reproduction on the 4090: all 7 scenes via reproduceMain.
  JAX MAE mean 1.44 m (matches paper ~1.4 m); IARPA mean 1.88 m. Env built end-to-end by
  01_setup_env.sh; every setup error folded back into the scripts (single self-installing
  path for the advisor). Phase-1 milestone complete. Earthdata auth verified (~/.netrc persisted); GEDI test query OK.
- **2026-06-22** — Cowork scaffolding session. Built git repo, scripts, configs, notebook.
- **2026-06-22** — Repo initialized; planning docs added. No code run yet.
