# STATUS — EOGS++ extensions (Paper 1 pivot) — living notebook

**Convention:** read FIRST each session; UPDATE at the END; commit + push. Companion to the
original STATUS.md (canopy/lidar work, now CLOSED — see its final sections for why) and HANDOFF.md.
**Last updated:** 2026-06-24.

---

## 0. Why we pivoted (one paragraph)
The canopy/under-canopy-lidar direction was CLOSED after a control showed GEDI-ground interpolation
(0.9 m) beats the optical fusion (9.1 m) on flat terrain, and the only terrain where interpolation
fails (steep forest) is exactly where GEDI ground is unreliable (slope + canopy penetration).
Decision brief: Paper1_Decision_Brief.docx. LESSON (now a standing rule): run the idea-killing
trivial baseline BEFORE building. We now pivot to extending EOGS++ (the new SOTA) where the optical
method is uniquely strong, not competing with a trivial lidar baseline.

## 1. New baseline: EOGS++ (= EOGS2, gardiens/EOGS2)
Paper: arXiv 2511.16542 (Bournez et al., 2025). Improvements over EOGS: internal optical-flow camera
refinement (no external BA), raw-PAN rendering (no pansharpen), opacity reset + early stopping, TSDF
post-processing. Reported ~1.33 -> 1.19 m building MAE vs EOGS. Env: conda eogsplus, python 3.9,
CUDA 11.8/12.1, 24 GB GPU; install.sh; data.zip (PAN + MSI + pansharpen + rpc_ba/rpc_raw + truth +
splits). 4090 OK. STATED LIMITATIONS we can exploit: (a) DROPPED MSI (joint MSI+PAN training
degraded -> aliasing); (b) global opacity-reset/shadow penalties DESTROY fine vegetation to sharpen
buildings (their words: JAX_260 vegetation not preserved); (c) TSDF post-proc disconnected from
training.

## 2. The five candidate ideas (assessed)
| # | Idea | Novelty | Speed/effort | Success odds | Survives idea-killer? | Verdict |
|---|------|---------|--------------|--------------|-----------------------|---------|
| 1 | Macroscopic FP32 anchoring (DEM dual-coord) | med | high effort | low | NO (origin-shift solves it) | DROP |
| 2 | SAR-optical fusion splatting (SpaceNet 6) | HIGH | slow (CUDA+SAR+coreg) | med, risky | maybe NOT (coreg may fail) | high-ceiling alt |
| 3 | 2DGS + Mip + MSI/PAN fusion | med-high | CUDA-heavy | med | likely | backup |
| 4 | Radiometric/material decomposition (6S, BRDF) | high | CUDA + 6S | med | risky (2D corr may win geom) | backup |
| 5 | Semantic-aware class-conditional regularization | med(-high for EO) | FAST (loss-level, no CUDA) | HIGH | YES (2D mask can't reconstruct veg) | **RECOMMENDED** |

## 3. RECOMMENDATION: Idea #5, killer-test FIRST
Why #5: loss-function level (reuses our M7/M8 per-Gaussian patching; NO CUDA); data in hand
(DFC2019); EOGS ships tree masks (prototype WITHOUT SAM, same-day); targets a weakness EOGS++ authors
ADMIT; the trivial baseline (2D-mask vegetation) destroys vegetation so it cannot win a joint
buildings+vegetation metric.

WIN METRIC (define upfront, Paper-1 lesson): tree-pixel DSM MAE (vegetation preserved) AND building
DSM MAE (stays sharp), reported separately, on vegetated DFC2019 scenes (JAX_068, JAX_260).

IDEA-KILLER TEST (run FIRST, ~1 day, 4090): on a vegetated scene, compare
  (a) EOGS++ uniform regularization (baseline),
  (b) trivial control: black out vegetation in 2D, train EOGS++ (sharp buildings, NO veg),
  (c) ours: gate opacity-reset/shadow-entropy by the shipped tree mask (strict on buildings,
      relaxed/off on vegetation).
PASS if (c) lowers tree-pixel DSM MAE vs (a) while holding building MAE ~constant, and (b) cannot
produce a vegetation surface at all. If (c) doesn't beat (a) on the tree metric -> reconsider.

ALTERNATIVE if higher ceiling wanted: Idea #2 (SAR-optical), accepting slower/riskier CUDA work.

## 4. Next steps (ordered)
1. Install EOGS2/EOGS++ on the 4090 (conda eogsplus + install.sh + data.zip); fold fixes into a
   setup script (advisor reproducibility). Reproduce its building MAE on 1-2 scenes.
2. Confirm EOGS++ exposes the regularizers (opacity reset, shadow/entropy) in the Python training
   loop and that tree masks align to the train grid.
3. Run the Idea-#5 killer test (a)/(b)/(c) on JAX_068 + JAX_260; record tree-pixel + building MAE.
4. If PASS: add per-Gaussian semantic gating (start with tree mask; then SAM features); ablations;
   write up. If FAIL: pivot to #3 or #2.

## 5. Decision log
- 2026-06-24: pivot decided; EOGS2 to be installed; idea #5 recommended; killer test designed.

## 6. Reusable assets from the closed direction
EOGS env-build experience + scripts/01; DFC2019 data + scene-build (scripts/19-21); per-Gaussian
loss-patching approach (eogs_mods/apply_*.py); DSM eval + tree-mask filtering; 3DEP/GEDI tooling
(not needed now but kept). All directly transfer to EOGS++ work.

---

## 7. Findings from pre-build review (2026-06-24)

MECHANISM (verified in EOGS source): vegetation is destroyed by L_opacity = opacity.sum()/N
(w_L_opacity=0.10, always on) pushing all opacities DOWN, then prune of opacity-logit < -6.0 deletes
the now-transparent (semi-translucent = fine vegetation) Gaussians. Buildings (opaque) survive.
=> testable with ZERO code change: w_L_opacity is a CLI arg. Scripts 25 (tree-vs-building DSM MAE)
+ 26 (ablation runner) built.

NOVELTY CAVEAT (literature): semantic-aware GS with vegetation/building classes already exists
(SA-GS, arXiv 2405.16923); satellite-3DGS regularization active (ShadowGS 2601.00939, EOGS++).
Idea #5's defensible novelty is NARROW (EO-specific class-conditional opacity/shadow regularizers)
-> solid-but-incremental, not landmark.

FAILURE MODES for #5 (all tested by Test 0): premise false (w=0 doesn't help veg); no tradeoff
(helps veg, doesn't hurt buildings -> just turn off globally); backfire (w=0 adds floaters, hurts
all); wrong metric (DSM MAE blind to fine-structure loss -> check view-PSNR); poor tree masks (95%
on JAX_068); win too small.

## 8. SHARPER FRAMING / better-idea recommendation
META-LESSON from the lidar death: choose a problem with STANDARD ground truth + PUBLISHED SOTA
baseline + clear novelty, so a win is unambiguous. On DFC2019 that = DSM MAE vs airborne lidar,
beating EOGS++.
- #5 vegetation-aware regularization: FAST probe, modest ceiling, thin novelty. TEST NOW (scripts 25/26).
- **2DGS-for-satellite (RECOMMENDED higher ceiling):** replace EOGS volumetric-Gaussian + disconnected
  TSDF with 2D surfels (native surface, normal-consistency, DSM straight off disks). NOVEL for
  affine/RPC satellite cameras; competes on the STANDARD DSM-MAE benchmark vs published SOTA = clean,
  degree-worthy win. Cost: CUDA (affine ray-disk intersection); killer test = gradients stable at
  grazing angles.
- MSI/spectral (Paper 2) + radiometric: high novelty but WEAK ground truth (spectral/material) = the
  same validation trap that killed lidar; pursue only with a validation plan.
- SAR fusion: highest novelty, slowest/riskiest (CUDA + SAR + coreg).
PLAN: run #5 precondition (scripts/26) FIRST (~1 day, nearly free). If real tradeoff -> quick win;
else commit to 2DGS-for-satellite as primary. Then install EOGS++ as the up-to-date baseline.

## 9. Decision rule (standing)
Before building ANY idea: (1) name the standard, ground-truthed metric; (2) name the published
baseline; (3) name the trivial baseline that could kill it and RUN it first; (4) only build if (3)
cannot already win. Applies to every candidate above.

## 10. Idea #5 KILLED by precondition (2026-06-25, JAX_214)

| w_L_opacity | VEG MAE | BLDG MAE | overall |
|-------------|---------|----------|---------|
| 0.10 (default) | 1.586 m | 2.796 m | 1.722 m |
| 0.0 (relaxed)  | 1.751 m | 2.777 m | 1.867 m |

Lowering the opacity penalty made VEGETATION WORSE (+0.165 m) and buildings ~unchanged (-0.019 m,
noise). OPPOSITE of the premise: the penalty HELPS vegetation (removes floaters), it does not destroy
it. No building-vs-vegetation tradeoff on the DSM metric => class-conditional regularization has
nothing to gain. The brief's "regularization destroys vegetation" is false for DSM accuracy (likely a
visual/appearance observation, which would be a weak metric and we LOSE on geometry). VERDICT: #5 not
worth a paper, not even a small one (negative ablation = at most a table row in a larger work). Cost:
~1 h. Test-before-build worked again.

## 11. DECISION: pivot to 2DGS-for-satellite as the primary
Rationale: only candidate with a STANDARD ground-truthed metric (DSM MAE vs airborne lidar on
DFC2019), a PUBLISHED SOTA baseline (EOGS++), and clear novelty (2D Gaussian surfels + normal
consistency under the AFFINE/RPC satellite projection; native DSM, drop TSDF). A win is unambiguous
and publishable. Next: (1) install EOGS++ as the baseline; (2) literature-check 2DGS-for-satellite
novelty; (3) the 2DGS killer test from idea #3 (affine ray-disk intersection: gradients stable at
grazing angles on a dummy disk scene) BEFORE committing CUDA effort; (4) if stable, build + benchmark
DSM MAE vs EOGS++. Several-papers goal best served by clean metric wins (2DGS geometry; Paper 2
spectral with a validation plan; Paper 3 uncertainty), NOT salvaged micro-ideas.

## 12. 2DGS-for-satellite: novelty, science, staged de-risk plan (2026-06-25)

NOVELTY (literature): a true 2D-SURFEL representation for satellite (RPC/affine projection -> DSM)
appears OPEN — EOGS, EOGS++, RPC-GS (2026, native RPC), SA-GS, SkySplat all use 3D Gaussians; 2DGS
surfels appear only for perspective cameras. Encouraging BUT hot/crowded area (RPC-GS is 2026) =>
real scoop risk, move fast. Refs: RPC-GS arXiv 2606.06690, SA-GS (ISPRS 2026), EGGS 2512.02932.

"ALL ANGLES" (what 2DGS could also solve): native surface DSM straight off the disks (DROP the
disconnected TSDF post-proc EOGS++ uses); fewer floaters via surface constraint (replaces opacity
hacks); sharper building edges (normal consistency); direct mesh extraction. Possible later combos:
+ Mip/MSI (idea #3 other half) and + native RPC (RPC-GS axis) for a larger contribution.

SCIENCE RISKS / FAIL MODES (capture early):
- Surfel prior may NOT help sparse oblique satellite views (Test A below).
- Affine/RPC ray-disk intersection may be UNSTABLE at grazing/oblique angles -> NaN gradients
  (Test B, before any CUDA).
- 2DGS may not BEAT EOGS++ even if stable (sparse ~20 oblique views) -> need early baseline compare.
- Scoop risk (fast area); RPC-GS argues native RPC > affine (our affine-2DGS could be one-upped).

STAGED PLAN (each gate before the next; capture fails early):
A. FREE precondition (scripts/27_erank_ablation.sh): EOGS's L_erank pushes Gaussians toward 2D
   disks (surfel prior, default OFF). Sweep w_L_erank; if flattening LOWERS DSM MAE vs the 1.722
   baseline -> 2DGS premise supported (necessary-not-sufficient). If it RAISES it -> early FAIL.
B. Killer test (small code, NO full CUDA yet): implement just the affine ray-disk intersection +
   backward on a dummy disk scene; check gradients are finite at grazing angles. If NaN -> the
   intersection math must be reformulated before committing.
C. Install EOGS++ (the SOTA baseline); reproduce its DSM MAE on 1-2 scenes.
D. Build the full affine-2DGS rasterizer; benchmark DSM MAE vs EOGS++ on DFC2019. Win = paper.

DECISION-RULE CHECK (section 9): metric = DSM MAE vs airborne lidar (standard, ground-truthed);
baseline = EOGS++ (published SOTA); trivial baseline = EOGS++ itself / 3DGS+TSDF -> we must beat it.
All satisfied. Proceed through gates A->D.

## 13. Gate A result + EOGS++ install (2026-06-25)

GATE A (erank/surfel-prior ablation, JAX_214): surfel prior HURTS DSM, monotonically.
| w_L_erank | VEG | BLDG | overall |
|-----------|-----|------|---------|
| 0.0 (baseline) | 1.586 | 2.796 | 1.722 |
| 0.1 | 1.613 | 2.997 | 1.769 |
| 0.5 | 1.645 | 2.838 | 1.780 |
=> YELLOW FLAG for 2DGS on satellite: sparse oblique views seem to benefit from 3D volumetric
flexibility; pure surfels remove it. CAVEAT: weak proxy — L_erank flattens primitives but keeps
EOGS's 3D alpha-blending; real 2DGS replaces the rasterization (ray-splat + normal/depth losses),
where its accuracy comes from. So NOT a definitive kill, but the bar for committing CUDA weeks is
higher (yellow flag + crowded/scoop-prone area + must beat well-tuned EOGS++). Definitive test = the
affine ray-disk gradient killer test (gate B), then a prototype.

PATTERN NOTE: cheap proxies keep returning negative (lidar, #5, erank) => EOGS is well-tuned;
beating it on DSM geometry is HARD. Bigger opportunities may be NEW capabilities (spectral/MSI,
uncertainty) where EOGS++ doesn't compete — but those need a validation-truth plan (the lidar trap).
Revisit the idea choice AFTER EOGS++ is up and we can read its code for real openings.

INSTALL: scripts/eogs2/install_eogs2.sh — clones gardiens/EOGS2, builds env eogsplus (py3.9) +
CUDA kernels (reusing our system CUDA 12.1 + gcc-12 + arch 8.9), downloads data.zip, preps affine
cameras. Smoke test: full_eval_pan.py ... scene=JAX_068. NEXT after install: reproduce EOGS++ DSM
MAE on 1-2 scenes; inspect its TSDF + dropped-MSI handling for the best injection point; then decide
2DGS gate B vs another idea.

## 14. EOGS++ reproduced + strategic redirect (2026-06-25)

REPRODUCTION (JAX_068): raw rendered DSM MAE 0.93 m (beats our EOGS 1.10); TSDF DSM 0.98 m
(WORSE than raw); TSDF-no-tree 0.93 m. KEY: TSDF post-proc does NOT improve DSM MAE here -> the
"replace TSDF with native 2DGS surface" motivation is weak (raw geometry already 0.93). Combined
with the erank yellow flag (surfels hurt), 2DGS is DE-PRIORITIZED.

MSI finding (paper S5.4): joint MSI+PAN training ("linear combination") DEGRADES geometry ~0.5 m;
authors drop MSI, conclude "process modalities separately." Cause UNDIAGNOSED (brief's "aliasing"
is a guess). They chased GEOMETRY only; never pursued MSI as a SPECTRAL PRODUCT. => MSI-for-geometry
is a stated dead end; MSI-for-spectral is unexplored.

META-PATTERN (important): EOGS++ geometry is SATURATED (~0.93 m); TSDF doesn't help, surfels hurt,
joint-MSI hurts. Every geometry probe negative; a TRIVIAL BASELINE keeps winning (interpolation,
2D-mask, PAN-only, likely pansharpen). Fighting for sub-0.93 DSM is low-yield in this mature/crowded
area. REDIRECT to CAPABILITY-GAP directions with CLEAN truth + WEAK trivial baseline.

CANDIDATES (clean-truth, capability-gap):
- **Calibrated DSM UNCERTAINTY (Paper 3) — RECOMMENDED.** No satellite-GS method outputs per-pixel
  uncertainty. Truth clean (calibration vs lidar error; ECE/reliability). Trivial baselines weak
  (photometric residual, opacity). Likely NO CUDA. NASA-relevant. Cheap precondition: is a simple
  uncertainty proxy from EXISTING EOGS outputs already well-calibrated vs lidar? If not -> opening.
- Spectral 3D from MSI (Paper 2): new product; clean held-out-MSI novel-view truth (no trap); BUT
  risks losing to 2D-pansharpen-then-splat; could read as "pansharpening in 3D."

DECISION NEEDED: uncertainty (cleaner, weaker baseline) vs spectral (newer product, baseline risk).
2DGS shelved unless a building-heavy scene shows TSDF actually helping.

## 15. DIRECTION CHOSEN: calibrated uncertainty for satellite GS (2026-06-25)

NOVELTY: general 3DGS uncertainty + NBV is crowded (SA-ResGS, FisherRF, BayesRays, view-dependent
uncertainty 2504.07370); for SATELLITE it appears OPEN — no lidar-calibrated DSM uncertainty, no
uncertainty-driven multi-date image selection. Core machinery adapted (FisherRF/BayesRays/ensemble);
defensible novelty = satellite adaptation + DSM-vs-lidar calibration + image-selection application
(which acquisitions to use/buy; valuable for expensive tasking). = Paper 3 of the thesis plan.

KEY STRUCTURAL ADVANTAGE over prior dead-ends: a SIMPLE signal that works is a WIN here, not a kill.
For geometry/DTM a strong trivial baseline = no contribution; for uncertainty, a well-calibrated
simple signal (e.g., multi-view depth disagreement) IS the contribution (first calibrated satellite-GS
uncertainty). More robust to the trivial-baseline trap.

METRICS (clean truth): (1) calibration of predicted sigma vs |DSM - lidar| (Spearman, ECE,
reliability diagram, AUSE/sparsification); (2) application: DSM MAE vs #images for uncertainty-guided
vs random/coverage selection. Both ground-truthed by lidar + held-out images.

STAGED PLAN: U0 precondition (correlate candidate per-pixel uncertainty proxies — multi-view DSM
disagreement, photometric residual, accumulated opacity, DSM roughness — with true error on EXISTING
outputs; informative signal => foundation). U1 calibrate best signal (ECE/AUSE). U2 uncertainty-guided
image selection (accuracy-vs-#images vs random). NEXT: inspect EOGS2 per-view render outputs to build
U0 against the real structure.

## 16. U0 precondition built (2026-06-25)
EOGS2 saves per-view georeferenced DSMs (UTM GeoTIFF, own grid each) for BOTH train+test
(eogsplus.yaml skip_train=False) -> ~19-view ensemble already on disk, no re-render. scripts/30
reprojects all to the lidar grid, computes per-pixel multi-view DISAGREEMENT (std) = uncertainty,
registers the mean to GT, and correlates uncertainty with |DSM-lidar| via Spearman + sparsification
(AUSE), with DSM-roughness as a baseline, split by tree/building. Decisive read: disagreement
Spearman rho >> 0 and low AUSE => informative uncertainty => foundation for U1 (calibrate) + U2
(uncertainty-guided image selection). rho ~ 0 => no signal, reconsider. Run on JAX_068 (iters 14200).

## 17. U0 RESULT — foundation POSITIVE, with a calibration-vs-correlation nuance (2026-06-25)

JAX_068, 19-view ensemble, vs lidar (overall MAE 1.095 m after +5.66 m register):
| signal | Spearman rho | AUSE |
|--------|--------------|------|
| multi-view disagreement | +0.589 | 0.185 |
| DSM roughness (baseline) | +0.617 | 0.178 |

=> Uncertainty IS predictable (rho~0.6) -> FOUNDATION CONFIRMED (first positive precondition since
the pivot). CAVEAT: trivial roughness MATCHES disagreement on CORRELATION. Does NOT kill it because:
(1) correlation != CALIBRATION — contribution is a calibrated sigma in METERS (ECE/reliability;
roughness is uncalibrated/arbitrary units), ideally a COMBINED estimator (disagreement + roughness +
photometric residual capture different things) beating any single signal; (2) roughness CANNOT pick
views — the headline contribution is the APPLICATION (uncertainty-guided acquisition/image selection),
which needs per-view epistemic uncertainty (disagreement), not roughness.

PLAN: U1 = calibrated multi-signal per-pixel uncertainty across scenes (CHEAP, no retraining; all
signals from existing renders; report ECE/AUSE beating single-signal baselines). U2 = uncertainty-
guided image selection (DSM MAE vs #images, guided vs random) = HEADLINE; requires retraining
experiments (~hours compute) -> justified now that U0 foundation holds. NEXT: extend script 30 with
photometric-residual + opacity + view-count signals and a held-out calibration check; then design the
subset-selection experiment.

## 18. Multi-scene reproduction running (2026-06-25)
reproduce_main.sh runs baseline EOGS + EOGS++ per scene. IARPA_001: EOGS++ rdsm 1.55 (TSDF 1.69 =
WORSE), baseline EOGS rdsm 1.57 (TSDF 1.59 = worse). => TSDF consistently does NOT improve DSM MAE
(now JAX + IARPA) -> 2DGS-replaces-TSDF stays buried; EOGS++ ~ baseline on IARPA_001 (gains scene-
dependent). This run produces the per-view ensembles for ALL 7 scenes (needed for U1 cross-scene
calibration + U2 baseline). scripts/31_uncertainty_batch.sh sweeps U0 across all scenes (auto-detects
scene + early-stopped iters) -> per-scene rho/AUSE table. Consistent rho>0 = solid foundation -> then
U1 (calibrated multi-signal) + U2 (uncertainty-guided image selection, the headline).

## 19. Next-stage code built while reproduction runs (2026-06-25)
- scripts/32_uncertainty_calibration.py (U1): multi-signal (disagreement+roughness+view-count)
  per-pixel uncertainty with LEAVE-ONE-SCENE-OUT calibration; reports Spearman/AUSE/ECE(m) per scene
  for each signal + the COMBINED estimator. Runs on the per-view DSMs the reproduction generates.
  Headline check: COMBINED beats single signals on AUSE and is well-calibrated (low ECE in meters).
- scripts/34_selection_precondition.sh (U2 precondition): trains EOGS++ on N random K-image subsets
  (via train.txt, NO code change) and reports the DSM-MAE SPREAD. Large spread => image selection has
  leverage -> build the guided selector. Small spread => U2 weak. FIRST RUN = verify output-dir naming
  + MAE capture (config reuse overwrites the same output dir, which is fine — MAE is grepped from the
  per-run log). 
- scripts/31 batch U0 already built (per-scene rho/AUSE once canonical runs exist).
SEQUENCE when 4090 frees: scripts/31 (confirm foundation across scenes) -> scripts/32 (U1 calibration)
-> scripts/34 (does selection matter?) -> if yes, build full uncertainty-guided selector (U2 headline).

## 20. Overnight run launched + glob bugfix (2026-06-25 night)
Have canonical runs: IARPA_001/002, JAX_004/068. Overnight pipeline (scripts/36) launched via nohup
(log -> results/overnight.log): trains missing JAX_214/260 + IARPA_003, then U0 batch (31) -> U1
calib (32) -> U2 precondition (34, JAX_068). BUG found+fixed before step 2: scripts/30+32 globbed
dsm/{scene}_*_pan.iio which found 0 for IARPA (different per-view DSM naming); now glob dsm/*.iio
minus Nadir/msi (robust) + print dsm-dir sample if <3 views. Picked up automatically by tonight's
steps 2-3. Foundation so far: JAX_068 rho 0.589, JAX_004 rho 0.461 (both positive; roughness ~matches
disagreement, as noted -> contribution = calibration + selection, not raw correlation). Read the log
tomorrow: per-scene rho/AUSE (31), COMBINED-vs-single calibration ECE/AUSE (32), subset MAE spread (34).
