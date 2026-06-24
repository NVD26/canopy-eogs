# STATUS — 3DGS for Earth Observation PhD (living lab notebook)

**Convention:** Read this FIRST at the start of every session. UPDATE it at the END
of every session, then `git commit` + `git push`. This file is the shared brain
between the laptop and the 4090 PC. Keep it honest — record failures, not just wins.

**Project:** Canopy-aware + spectral 3D Gaussian Splatting for Earth observation (UAH PhD).
**Active machine for this entry:** 4090 home PC.
**Last updated:** 2026-06-22.

---

## 0. Current focus (one sentence)

Paper-1 (canopy-aware EOGS): METHOD BUILT + TRAINING. GS-native two-surface model (per-Gaussian
ground height + 2nd render pass) is implemented as a reproducible patch and training on JAX_113
(21 views, ~46 GEDI ground anchors/view). Plumbing verified (top DSM unchanged at w=0); ground
fits GEDI (|ground-GEDI| -> cm) after the "detach top" identifiability fix. Next: score the ground
DSM vs 3DEP over the full extent, pool across JAX tiles, add uncertainty, write the paper.

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
| M6 | Paper-1 design doc + non-technical progress report written | ✅ |
| M7 | Lidar canopy-top loss wired into EOGS; anchor->view projection verified (0.3 px vs RPC) | ✅ |
| — | **Result:** canopy-top supervision MONOTONICALLY hurts DSM (clean negative control) | ✅ |
| M8a | Under-canopy PROBLEM quantified vs 3DEP: surface ~15 m above true ground under tall canopy | ✅ |
| M8b | Denser scene built from DFC2019 source (JAX_113: 21 views, DSM 1.34 m; ~46 anchors/view) | ✅ |
| M8c | GS-native two-surface model (per-Gaussian ground + 2nd render pass) implemented as a patch | ✅ |
| M8d | Two-surface trained on JAX_113: ground fits GEDI, top preserved (after detach-top fix) | ◐ running |
| M9 | Ground DSM scored vs 3DEP bare-earth over full extent; pooled across the 14 anchor JAX tiles | ☐ |
| M10 | Calibrated per-pixel UNCERTAINTY head on canopy height + ground | ☐ |
| M11 | Ablations + baselines + figures + tables (camera-ready evidence) | ☐ |
| M12 | Paper write-up -> arXiv preprint, then journal/conference submission | ☐ |

**PHASE-1 COMPLETE (EOGS baseline). PHASE-2 (canopy-aware method) IN PROGRESS — M7 negative
control done, problem quantified, two-surface method built and training.**

---

## 2. Done so far

- 2026-06-22: Repo scaffolding created. Verified EOGS code repo bundles DFC2019 +
  IARPA tiles + DSM truth (`dataset_v01`).
- 2026-06-22: Full environment built on the 4090 via `scripts/01_setup_env.sh`
  (miniconda, conda ToS, apt build tools, torch cu121, EOGS, CUDA 12.1 toolkit, kernels).
- 2026-06-22: EOGS reproduced on all 7 scenes (4 JAX + 3 IARPA); DSM MAE matches paper.

## 3. In progress right now

- M8d: two-surface model TRAINING on JAX_113 (w_L_ground=1.0). Ground render adds a 2nd pass ->
  ~2 it/s. Watching: ground fits GEDI (|ground-GEDI| -> cm), top DSM stays ~1.3 m (separation).
- After this run: render the GROUND top-down and score vs 3DEP over the full extent (M9).

## 4. Next steps (ordered) — plan through paper submission

1. (M8d, now) Finish the JAX_113 two-surface run; confirm ground fits GEDI + top preserved.
2. (M9) GROUND vs 3DEP, the headline number: render the learned ground top-down, convert to NAVD88,
   score MAE vs 3DEP bare-earth on tall-canopy pixels. Target: bring the ~15 m surface error down to
   ~1-3 m. Use a HELD-OUT anchor split (supervise on 80%, evaluate on 20% + dense 3DEP) so the number
   reflects GENERALIZATION, not memorization.
3. (M9, pooled) Build the other 13 anchor-bearing JAX tiles (scripts/21 loop), train two-surface on
   each, POOL the ground-vs-3DEP evaluation (~56+ footprints / wider area) for statistical weight.
4. (M10) Add a calibrated uncertainty head (predict per-pixel sigma on canopy height + ground);
   report calibration (e.g., reliability curves, % within k-sigma).
5. (M11) Ablations: no-lidar baseline, canopy-top-only (the negative control), ground-only, both;
   sensitivity to anchor density + weight; compare to EOGS surface-as-ground and to plain GEDI
   interpolation. Figures: surface vs ground vs 3DEP maps; error maps; dose-response; per-tile table.
6. (M12) Write-up: intro/problem (the 15 m gap), method (two-surface GS + spaceborne-lidar ground
   supervision + uncertainty), experiments (JAX, pooled), results, ablations, limitations (anchor
   sparsity, single-site). Target an arXiv preprint first, then a remote-sensing venue (e.g., ISPRS
   J. Photogrammetry & Remote Sensing / IEEE TGRS / CVPR EarthVision workshop).

### Larger-scene upgrade (parallel / if time): full WorldView strips
DFC2019 RGB are per-tile 256 m crops (sparse, ~4-10 anchors/tile in the truth grid, though the IMAGE
extent sees ~46/view). For a single dense ~1 km scene (~166 anchors) we would acquire the original
CORE3D/MVS3DM WorldView-3 strips and crop a custom AOI (centre -81.6772, 30.3464). Documented in
docs/M8_large_scene_plan.md. Not required for a first result; strengthens the flagship scene.

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
- **Anchor builder UPGRADED + RUN (scripts/12 now uses 3DEP for datum-align + outlier reject):**
  exact 256 m JAX_068 tile = 7 quality -> **4 clean** anchors (LOW-COUNT). 2 km box = 1356 -> **1315
  clean** anchors (VALIDATED). Per-area datum -29.8 m (matches wide validation), residual ~0.4 m,
  canopy heights up to 53 m. Output: data/anchors/<scene>_gedi_anchors.npz (UTM E,N + ground &
  canopy-top in NAVD88/3DEP frame). DECISION NEEDED: a single DFC2019 tile is too anchor-poor to
  supervise; options = (a) supervise over the full ~600 m image footprint per tile (~tens of
  anchors), (b) pool the 4 JAX tiles, (c) build larger custom EOGS scenes over Jacksonville with
  open imagery (1000s of anchors). Reading EOGS loader to choose + wire M7.
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

- **2026-06-22** — M6 anchors. Upgraded scripts/12 to a TRUSTED anchor builder: fetches 3DEP DTM,
  uses it ONLY for outlier rejection (drop ~10% blunders), and saves RAW GEDI ellipsoidal heights
  (ground + canopy top) in UTM. KEY CORRECTION found by reading EOGS source: EOGS reconstructs in
  the satellite RPC frame = WGS84 ellipsoid, same as GEDI; 3DEP is NAVD88, so the -29.8 m geoid
  offset is for 3DEP-eval ONLY and must NOT be applied to supervision heights (would inject ~30 m
  bias). Counts: exact 256 m tile = 4 clean anchors (too few); 2 km box = 1315 clean (validated).
  EOGS world frame learned: normalized UTM, world=(UTM-centerofscene_UTM)/scale; camera affine maps
  world xyz -> (u,v,altitude); anchors project via that. OPEN DECISION: supervise per-tile (anchor-
  poor) vs larger custom Jacksonville scene (anchor-rich). Next: M7 lidar loss + anchor->view projector.

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

---

## 8. M7 — lidar canopy-anchor loss IMPLEMENTED (2026-06-23)

**Projection PROVEN (scripts/14, JAX_068, 2 km anchors):** height round-trip 0.000 m
(centre/scale exact); affine-vs-RPC pixel disagreement median 0.31 px / 95th 0.43 px
(< 2 px gate) — our anchor projection matches the satellite RPC. 19 views, ~4 anchors
in view each (78 anchor-views). VERDICT: PASS.

**Loss wired as a reproducible PATCH (not edits to a pinned clone):**
- `eogs_mods/lidar_anchors.py` — builds a per-view index {image_name: u,v (NDC), canopytop,
  ground}, using the verified projection `world=(UTM-centre)/scale; uva=world@A.T+b`.
- `eogs_mods/apply_paper1_lidar.py` — idempotent patcher; 6 unique-marker edits to the EOGS
  clone (ModelParams.lidar_anchors_path; OptimizationParams.{w,iterstart}_L_lidar; train.py:
  build index after Scene, init L_lidar, compute it by grid_sample of altitude_render at the
  footprint pixels vs GEDI canopy-top, add `w_L_lidar*L_lidar` to the total loss). Prints a
  loud WARNING if anchor names match no train camera (guards silent zero-supervision).
- `scripts/15_train_lidar.sh` — applies the patch, then runs baseline (w=0) vs lidar (w=0.1)
  on the SAME scene/seed; each: train -> render -> eval DSM MAE overall AND tree-pixels
  (--filter_tree). Same code path; only w_L_lidar differs.

Validated OFFLINE (no GPU): patcher applies cleanly to a fresh EOGS clone, is idempotent,
both new args present, patched train.py/arguments compile; anchor .npz keys (E,N,ground,
canopytop) match the loader; heights ellipsoidal (ground ~-21 m, top ~-2.5 m, ~18 m canopy).

NEXT (on the 4090): `bash scripts/15_train_lidar.sh JAX_068 0.1` -> record both MAE numbers
here. Expectation: small effect (only ~4-7 unique anchors/scene); this is the plumbing test.
If the loss behaves (L_lidar logged, decreasing; no NaN; tree-MAE moves), scale up anchors
via a larger custom Jacksonville scene before drawing conclusions. Then M8 = two-surface
(ground + canopy) decomposition using the `ground` anchors already in the .npz.

### M7 first RESULT (2026-06-23, JAX_068, controlled same-seed ablation)

| Condition | Overall DSM MAE | Tree-pixel MAE |
|-----------|-----------------|----------------|
| Baseline (w_L_lidar=0)      | 1.101 m | 1.076 m |
| + GEDI canopy-top loss (w=0.1) | 1.219 m | 1.201 m |

Supervision was confirmed ACTIVE: `[P1-lidar] 78 anchor-views across 19 cameras; 17 match
train-cams`. Same scene/seed/init; only w_L_lidar differs (seed noise here ~0.002 m), so the
+0.118 m (overall) / +0.125 m (tree) degradation is REAL and attributable to the loss.

INTERPRETATION (expected, not a bug): (1) EOGS's optical surface is already accurate at the
CANOPY TOP — where multi-view stereo works — so anchoring it to GEDI canopy-top can only match
or harm; (2) GEDI "canopy-top" is the highest return in a ~25 m footprint while we sample a
single 0.5 m pixel, so the target is biased high + noisy vs a point. CONCLUSION: canopy-top is
where optical already wins; spaceborne lidar's value is the HIDDEN GROUND. This MOTIVATES M8
(two-surface ground supervision) — the actual Paper-1 contribution. Negative control = good.

DIAGNOSTIC QUEUED: scripts/16_lidar_weight_sweep.sh (w in {0,0.01,0.03,0.1,0.3}) to confirm a
monotonic dose-response (loss active but target unhelpful). Then M8.

### M7 weight sweep (2026-06-23, JAX_068) — clean monotonic negative control

| w_L_lidar | Overall MAE (m) | Tree MAE (m) |
|-----------|-----------------|--------------|
| 0    | 1.108 | 1.083 |
| 0.01 | 1.122 | 1.097 |
| 0.03 | 1.153 | 1.127 |
| 0.10 | 1.250 | 1.253 |
| 0.30 | 1.336 | 1.315 |

Monotonic in both columns -> loss is active and the canopy-top TARGET (not a bug) drives the
degradation. PAPER FINDING: GEDI canopy-top supervision does not help optical EOGS (optical MVS
already resolves the top); spaceborne lidar's value is the under-canopy GROUND. -> Build M8.
CSV: results/lidar_weight_sweep_JAX_068_*.csv

### M8-B premise QUANTIFIED on real data (2026-06-23, JAX_068, datum-validated)

All surfaces in NAVD88 (EOGS/GT DSM are ellipsoidal -> -geoid(-29.84); GEDI ground likewise).
Datum+placement check: GEDI ground vs 3DEP at anchor cells = 0.27 m abs-median (PASS).

| Measurement (tree/under-canopy) | Value |
|---------------------------------|-------|
| EOGS surface vs GT airborne DSM (tree)        | MAE 1.08 m  (EOGS gets the SURFACE right) |
| GT airborne DSM vs 3DEP bare-earth (tree)     | median +5.68 m (= canopy height) |
| EOGS surface vs 3DEP bare-earth (tree)        | MAE 9.08 m  (surface != ground) |
| Tall-canopy pixels (airborne canopy > 3 m)    | 138,211 px (53% of tile), median canopy +11.3 m |
| **EOGS surface vs 3DEP on TALL canopy**       | **MAE 14.93 m**  (the under-canopy headroom) |

PAPER RESULT (the problem, measured): EOGS reconstructs the optical SURFACE to ~1 m, but under
tall canopy that surface is ~15 m ABOVE the true ground. Spaceborne-lidar ground supervision is
needed to recover it. CONFIRMED WALL: only 4 GEDI ground anchors inside the 256 m DFC2019 tile
-> cannot fit a ground DTM per-tile. FIX = larger custom Jacksonville scene (the 2 km AOI holds
1315 validated anchors). Script: scripts/17_ground_dtm_premise.py (--dry-run for the stats).

### M8 larger-scene AOI selected (2026-06-23)

DFC2019 256 m tiles are anchor-starved (4 anchors). A larger custom Jacksonville scene fixes it.
GEDI density over the validated 2 km AOI (scripts/18_pick_large_aoi.py):

| AOI | GEDI ground anchors (all canopy>3 m) |
|-----|--------------------------------------|
| 512 m  | 59  |
| 768 m  | 101 |
| 1024 m | 166 |

CHOSEN: ~1 km AOI, centre UTM (434910, 3357367) / lon-lat (-81.6772, 30.3464), ~166 anchors.
Build route (Sat-NeRF lineage, EOGS-compatible): DFC2019 Track3-RGB + Track3-Truth ->
create_satellite_dataset.py (centreborelli/satnerf) WITH bundle_adjust (centreborelli/
sat-bundleadjust) for accurate RPCs -> EOGS scene (to_affine) -> train -> M8 ground supervision
+ eval vs 3DEP DTM over ~166 anchors. See docs/M8_large_scene_plan.md.

---

## 13. Handoff inventory (read this to resume cold)

**Repo:** github.com/NVD26/canopy-eogs (private). Source of truth = this file + git. Two machines:
4090 home PC (WSL2 Ubuntu, host LYRA) runs all GPU work; laptop plans. Conda envs: `eogs`
(EOGS training, torch cu121, system CUDA 12.1, gcc-12) and `ba` (scene building: bundle_adjust +
rasterio + fire; s2p intentionally omitted). EOGS clone: ~/eogs-src/EOGS (commit cca973e + our
patches). DFC2019 source: ~/eogs-src/DFC2019 (Track3-RGB-1 [JAX], Track3-RGB-2 [OMA],
Track3-Truth, Track3-Metadata). Built scenes: ~/eogs-src/scenes and EOGS/data/{rpcs,images,
affine_models,truth}/<tile>. Tools: ~/eogs-src/tools/{satnerf,sat-bundleadjust}.

**Scripts (scripts/):** 00 gpu check; 01 env build (all fixes folded in); 02 earthdata auth;
03 get EOGS data; 04 prep cameras; 05 run EOGS; 06 eval DSM MAE; 10 query GEDI/ICESat-2;
11 inspect data; 12 build GEDI anchors (3DEP datum-align + outlier reject; saves RAW ELLIPSOIDAL
heights + geoid_offset_to_navd88); 13 validate GEDI vs 3DEP; 14 anchor->view projection check
(PASS = <2 px vs RPC); 15 train+ablate canopy-top lidar loss; 16 lidar weight sweep;
17 ground-DTM premise (problem quantified vs 3DEP; --dry-run prints datum sanity);
18 pick larger AOI; 19 setup scene tools (ba env); 20 inspect DFC2019 tiles/coverage;
21 build one EOGS scene from a DFC2019 tile; 22 train two-surface (plumbing w=0, then w>0).

**Method patches (eogs_mods/):** lidar_anchors.py (per-view anchor index, target=canopytop|ground;
projection verified by scripts/14). apply_paper1_lidar.py (M7 canopy-top loss; NEGATIVE control).
apply_paper1_twosurface.py (M8 GS-NATIVE two-surface: per-Gaussian ground height a_ground =
a_top.detach() - softplus(g); 2nd render pass for ground map; GEDI ground loss + collapse prior;
adds args w_L_ground/iterstart_L_ground/w_L_groundcollapse/ground_lr). Both are idempotent, apply
to a fresh EOGS clone, validated offline (apply+compile) before GPU. Anchors reused region-wide:
data/anchors/JAX_068_gedi_anchors.npz (1315 anchors, 2 km AOI); symlinked per tile.

**Key numbers so far (all real data, datum-validated):** EOGS reproduced JAX mean 1.44 m.
GEDI ground vs 3DEP 0.27 m at anchors (datum -29.84 m = geoid). Canopy-top loss: monotonic harm
(negative control). Under-canopy problem: EOGS surface 14.9 m above 3DEP ground on tall canopy.
JAX_113 scene: 21 views, DSM 1.34 m, 977 anchor-views (~46/view). Two-surface: ground fits GEDI to
cm; top preserved after the detach fix (run in progress at last update).

**RESUME POINT:** finishing scripts/22 on JAX_113 (two-surface). Next = M9: render learned ground
top-down -> NAVD88 -> MAE vs 3DEP on tall-canopy pixels with a HELD-OUT anchor split (generalization,
not memorization); then pool across the 14 anchor JAX tiles; then uncertainty (M10), ablations
(M11), paper (M12). See section 4 for the ordered plan and docs/M8_large_scene_plan.md.

**SCIENCE-INTEGRITY RULES (do not break):** validate every step against independent ground truth
before trusting it; inspect real data/formats before coding; keep SYNTHETIC prototype results
separate from REAL-data results; report negative results honestly; use held-out splits for any
accuracy claim; every manual fix gets folded back into the scripts. Novelty = fusing spaceborne
lidar GROUND returns into fast Gaussian-splatting EO reconstruction to recover under-canopy terrain
with calibrated uncertainty; confirmed distinct from EOGS/EOGS++/Sat-NeRF/EO-NeRF (camera-only) and
from canopy-top-only supervision (which we showed fails).

### M8d UPDATE (2026-06-23): two-surface SEPARATION achieved (identifiability fix)

First two-surface runs COLLAPSED: ground loss dragged the shared geometry down, so both surfaces
sank to the laser ground (|top-GEDI|==|ground-GEDI|->0; TOP DSM degraded 1.32->2.9/3.05 m).
ROOT CAUSE: detaching only the altitude VALUE was insufficient — the ground render still
backprops through the shared Gaussian positions/opacities into _xyz.
FIX (scientific design decision): the ground is a PURE per-Gaussian vertical offset over FROZEN
geometry. Added `detach_geometry` to render(); the ground pass detaches means3D/opacity/scales/
rotations so L_ground trains ONLY g (a_ground = a_top.detach() - softplus(g)). Photometric owns
the top; lidar owns the offset.
RESULT (JAX_113, w_L_ground=1, 1500-iter check): surfaces DIVERGE — it1000 |ground-GEDI|=1.82 m vs
|top-GEDI|=5.85 m; it1500 1.95 vs 6.20 m. TOP DSM 3.63 m at 1500 iters is undertraining (vs 1.32 m
at 5000). Running full 5000 next. Patch: eogs_mods/apply_paper1_twosurface.py (idempotent +
self-correcting on already-patched clones).

### M8d/M9 RESULTS (2026-06-23, JAX_113, full 5000-iter two-surface run)

METHOD VALIDATED: surfaces separate cleanly. Final |ground-GEDI|=0.19 m (ground fits the laser),
|top-GEDI|=2.97 m (top held at canopy height), TOP DSM MAE=1.37 m (= baseline 1.34 -> optical top
preserved). The geometry-detach two-surface model recovers the hidden ground at supervised
footprints without harming the canopy reconstruction. Dump saved: <out>/twosurf_state.npz.

M9 top-down score (scripts/23) NOT YET TRUSTWORTHY -> two issues found:
(1) Per-Gaussian gridding is wrong: the learned ground lives in the ALPHA-COMPOSITE, not single
    Gaussians; the crude max-cell rasterization recovered only ~0.16 m of the ~2.8 m offset, and
    used just 38k/1.25M Gaussians. FIX = render the ground top-down via the renderer (patch
    render.py + save/load _ground), not gridding.
(2) REAL FINDING: training |top-GEDI|~3 m means our usable GEDI ground anchors sit under only ~3 m
    canopy; the dense tall-canopy pixels (~9 m) have almost no anchors. GEDI often fails to return a
    clean ground echo under dense tall canopy, so anchors are biased to shorter/sparser canopy.
    With independent per-Gaussian offsets (no spatial smoothing) the ground descends only at
    footprints. -> Need (a) spatial smoothness/TV on the ground offset for generalization, and
    (b) honest reporting of where GEDI provides ground (anchor canopy-height distribution).

NEXT: (i) render-based ground DSM for a valid top-down number; (ii) anchor canopy-height histogram
to quantify the GEDI ground-return bias; (iii) add a smoothness prior on the per-Gaussian ground
offset; (iv) re-evaluate generalization. The mechanism is proven; these make the result general
and the evaluation faithful.

### CORRECTION (2026-06-23): GEDI anchors are NOT short-canopy biased

Checked the anchor canopy heights directly (canopytop - ground): median 13.5 m, 81% > 8 m,
only 10% < 5 m. So the earlier hypothesis (anchors under ~3 m canopy) is WRONG. The anchors sit
under TALL canopy. The real puzzle: training |top-GEDI| ~ 3 m means the OPTICAL surface at the
ground-anchor projected pixel is only ~3 m above the laser ground, even though true canopy there
is ~13.5 m. Likely cause: where a ground return projects (at ground altitude) vs where the optical
canopy top sits differ (parallax/structure); the crude per-Gaussian grid score cannot resolve
this. ACTION: build the FAITHFUL top-down ground render (render.py, alpha-composite) and inspect
A_top vs A_ground vs 3DEP on the same grid before drawing any generalization conclusion. Method
mechanism remains validated (ground render fits GEDI 0.19 m; top preserved 1.37 m).

### M9 faithful score (2026-06-23, JAX_113, rendered ground DSM via patched render.py)

Datum: geoid -29.84 m, bare-align offset -0.57 m. Valid 6.28M px; tall-canopy (>3 m) 2.72M;
far-from-anchor tall 2.22M; median canopy 8.4 m. Score vs 3DEP bare-earth (NAVD88):

| (tall canopy) | all tall | far-from-anchor (generalization) |
|---------------|----------|----------------------------------|
| TOP as ground (baseline) | 11.78 m | 12.14 m |
| LEARNED ground (two-surface) | 11.05 m | 11.80 m |

HONEST FINDING: the model fits GEDI at footprints (training 0.19 m) but does NOT generalize — the
ground descends only ~0.7 m (all) / 0.34 m (far) from the canopy top. CAUSE: per-Gaussian offsets
are INDEPENDENT and the collapse prior pins unsupervised Gaussians to the top, so the recovered
ground stays at the canopy except at sparse footprints. The render-based DSM + datum align are now
trustworthy (this replaces the crude per-Gaussian grid score). NEXT METHOD STEP: add a SPATIAL
SMOOTHNESS prior so the footprint offset propagates (render-space TV on A_ground first — cheap;
then world-space kNN smoothness on g, or a smooth-field parameterization g=f(position) if needed),
and reduce the collapse weight. Re-train + re-score. Mechanism is proven; generalization is the
open problem and the next contribution.

### M9 with SMOOTHNESS (2026-06-23, JAX_113, render-space TV on ground, w_L_groundtv=0.1)

Render-space total-variation smoothness on the ground render => the per-Gaussian offset now
PROPAGATES between sparse footprints. Faithful render-based score vs 3DEP (NAVD88), tall canopy
(median 8.8 m), TOP DSM preserved at 1.32 m. Training: |ground-GEDI|->0.26 m, |top-GEDI| held ~2.9 m.

| (tall canopy) | all tall | far-from-anchor (generalization) |
|---------------|----------|----------------------------------|
| TOP as ground (baseline)        | 11.92 m | 12.19 m |
| LEARNED ground, NO smoothness   | 11.05 m | 11.80 m |
| LEARNED ground, WITH TV smooth  |  9.10 m |  9.64 m |

RESULT: smoothness enables GENERALIZATION -> far-from-anchor error drops 11.80 -> 9.64 m (2.55 m,
vs 0.34 m without). Ground recovered ~2.8 m of the ~12 m canopy gap, generalizing away from
footprints, top intact. POSITIVE result; the spatial prior is the right lever. NEXT: sweep
w_L_groundtv (more smoothing likely recovers more), then world-space coupling / smooth-field if
needed; pool JAX tiles; uncertainty (M10). Figure: results/m9_eval.png (after), m9_before saved.

### NEXT EXPERIMENTS QUEUED (2026-06-23) — priority order

1. INTERPOLATION CONTROL (highest value, CPU-only, no retrain): scripts/23 now reports 4 rows —
   TOP baseline, GEDI-ground interpolated (no optical), optical-TOP-minus-interpolated-canopy
   (strong control), and ours. The paper hinges on OURS beating both controls FAR FROM ANCHORS;
   if the strong control matches ours, the optical fusion may not add value (must know before writing).
   Run: python scripts/23_score_ground.py --model-path <latest exp> --tile JAX_113.
2. SMOOTHNESS SWEEP (GPU, deferred): runner now takes WGTV env (w_L_groundtv). Try WGTV=0.3, 1.0;
   watch far-from-anchor MAE down + TOP DSM stays ~1.3 m. Current best (WGTV=0.1): 9.10/9.64 m.
3. MULTI-SITE POOLING (GPU): build + train the other anchor-bearing JAX tiles (scripts/21 loop),
   score each, pool — required for publication (one site is not enough).
4. M10 uncertainty head; M11 ablation table; M12 paper.
Results to be pasted back here as each is run.

### CRITICAL FINDING (2026-06-23): interpolation control beats the method on flat terrain

scripts/23 controls (JAX_113, vs 3DEP, tall canopy):
| approach | all tall | far-from-anchor |
|----------|----------|-----------------|
| TOP baseline | 11.92 | 12.19 |
| GEDI ground INTERPOLATED (no optical) | **0.88** | **0.95** |
| optical TOP - interp canopy (strong control) | 7.94 | 8.77 |
| LEARNED ground (ours) | 9.10 | 9.64 |

VERDICT: on FLAT Jacksonville terrain, simply interpolating the sparse GEDI ground gives a 0.9 m
DTM — far better than ours (9.1 m) AND than the optical control (7.9 m). The optical fusion adds
NO value where terrain is smooth, because interpolation is trivially accurate (even 30 m from a
track). Two issues: (1) flat terrain => interpolation unbeatable; (2) our learned offset is worse
than directly interpolating the same offset (method inefficiency).

IMPLICATION (reframe, honest): the two-surface / lidar-fusion contribution can ONLY be justified
where GEDI is SPARSE RELATIVE TO TERRAIN VARIATION — forested terrain with RELIEF (hills/mountains,
valleys/ridges between laser tracks) where interpolation FAILS and the optical canopy structure
(which follows terrain) carries between-track information. Jacksonville = right place to BUILD/debug
the machinery, WRONG place to prove value. The control did its job: it prevented an unjustified
paper. NEXT DECISION: select a forested + high-relief AOI with GEDI + airborne-lidar (3DEP or
equivalent) coverage, re-run the control there; the method earns its place only if it beats
interpolation in that regime. Also fix method to at least match the optical control. Do NOT write
the paper on flat-terrain results.

### PROCESS RULE + SALVAGE PLAN (2026-06-23)

PROCESS RULE (standing): for ANY new idea, run the TRIVIAL baseline that would make the method
pointless BEFORE building the method. Here that was GEDI-ground interpolation; running it at the
premise stage would have cost ~1 h and saved the two-surface build. Validate worth-before-effort.

WHY relief is the only hope, and why the CURRENT method still wouldn't win there: our method learns
a per-Gaussian ground offset from GEDI and smooths it -> it uses the SAME information as
interpolation, propagated worse (lost 9.1 vs 0.88, and vs the optical control 7.94). For the optical
to add value: (a) GEDI interpolation must FAIL (only in relief, where terrain varies faster than
GEDI samples), AND (b) the method must extract terrain detail from the optical SURFACE -> reformulate
as ground = detailed-optical-surface - SMOOTH canopy-height (this is control B, 7.94), not
offset-smoothing. Value is a NARROW niche even if it works (GEDI is sub-meter where it lands).

PRECONDITION TEST (scripts/24_relief_precondition.py, CPU/3DEP only, no imagery): does GEDI-density
interpolation FAIL on relief terrain? Run on flat (Jacksonville, expect sub-m = no headroom) vs
relief (Smoky Mtns). Only if relief interpolation error is LARGE (>~3 m) is the optical version
worth building. If small everywhere -> direction dead.

SALVAGE (reuse all infra: EOGS repro, GEDI/ICESat-2 pipeline, 3DEP eval, two-surface renderer,
controls): if relief shows headroom -> build optical reconstruction over a relief forest + test the
reformulated method vs interpolation. If not -> pivot to a publishable SYSTEMATIC STUDY: "When does
spaceborne-lidar + optical fusion help under-canopy terrain mapping?" (controlled across terrain
ruggedness; the flat-terrain negative + the interpolation controls are the result). That salvages the
effort honestly. Either path reuses everything built so far.

### DECISION (2026-06-23): STOP the lidar-fusion under-canopy DTM direction

scripts/24 precondition results: Jacksonville (flat, relief 17 m) interpolation MAE 0.24 m = NO
headroom; Smoky Mtns (relief 1441 m, slope-std 22.4%) interpolation MAE 4.10 m = headroom EXISTS.
BUT the headroom coincides with GEDI's FAILURE MODE -> direction killed:
- On ~20deg slopes, a 25 m GEDI footprint spans ~9 m of ground elevation; GEDI ground RMSE rises
  from ~1 m (flat) to ~5-15 m (>20deg). The 4.1 m headroom is SMALLER than the GEDI ground noise
  there -> supervision noise > signal; cannot recover 4 m detail from +/-5-9 m anchors.
- Dense high-biomass forest (the relief case) is also where GEDI most often returns NO clean ground
  -> anchors noisier AND sparser exactly where needed. Catch-22, intrinsic to GEDI.
- Secondary: optical canopy-top encoding fine terrain is unproven; EOGS-on-forest unvalidated
  (our 1.3 m was flat + buildings); multi-view VHR over mountains is not in any open benchmark.
VERDICT: not worth the effort/resources. DO NOT build the relief version.

SALVAGE / PIVOT (take to advisor): (a) the infrastructure (EOGS repro, GEDI/ICESat-2 + 3DEP
pipeline, two-surface renderer, controls, scripts/24) is reusable; (b) honest short analysis is
publishable: "Spaceborne-lidar interpolation is a hard-to-beat baseline for under-canopy terrain;
optical fusion's only headroom (steep forest) coincides with GEDI's own failure mode" — a
cautionary/when-does-it-work study; (c) reconsider Paper 1's angle so 3DGS-for-EO competes where it
is uniquely strong (dense high-res structure/spectral) rather than against a trivial lidar baseline.
OPTIONAL hard-confirm before advisor: run scripts/13-style GEDI-vs-3DEP in the Smokies; predict GEDI
ground error ~ 4-9 m there (>= headroom), which would seal the decision.

### BRAINSTORM: can anything beat interpolation? (2026-06-23, for the record)

Fundamental barrier: optical CANNOT see ground under closed canopy (physics). Only canopy-penetrating
spaceborne sensors give ground = GEDI/ICESat-2 (sparse, slope-limited) + SAR (phase center in canopy,
not ground). So to beat interpolation a method must inject between-point ground info; only sources:
- Canopy-top as terrain proxy (our method): fails in steep/dense regime where needed. CLOSED.
- Leaf-off optical stereo (deciduous, e.g. Smokies): direct ground via bare-canopy stereo; GENUINE but
  occlusion/slopes limit, established photogrammetry (modest novelty), demotes GEDI to validation.
- Optical-informed SLOPE-AWARE GEDI retrieval: use optical slope to deconvolve GEDI within-footprint
  slope spread -> better ground per footprint -> interpolate corrected anchors. MOST novel, attacks the
  exact failure mode, but needs L1B waveforms + slope assumption; reframes as a lidar-retrieval paper.
- Learned cross-site canopy->terrain prior: physically weak relationship, large effort, uncertain.
VERDICT: for "optical+lidar fusion beats interpolation for under-canopy DTM" the door is CLOSED; the
only live cracks lead to DIFFERENT goals. Deliverable: Paper1_Decision_Brief.docx (advisor).
