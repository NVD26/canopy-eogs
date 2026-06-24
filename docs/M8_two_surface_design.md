# M8 — Two-surface (canopy + ground) design note

**Date:** 2026-06-23   **Scene under test:** JAX_068 (Jacksonville, FL; heavily vegetated)

## Why M8 (what M7 told us)
M7 wired a GEDI **canopy-top** anchor loss into EOGS. Controlled same-seed ablation +
weight sweep (JAX_068):

| w_L_lidar | Overall DSM MAE | Tree-pixel MAE |
|-----------|-----------------|----------------|
| 0    | 1.108 | 1.083 |
| 0.01 | 1.122 | 1.097 |
| 0.03 | 1.153 | 1.127 |
| 0.10 | 1.250 | 1.253 |
| 0.30 | 1.336 | 1.315 |

Monotonic degradation -> the loss is active and the **target** is the issue, not a bug.
Reason: EOGS's optical multi-view-stereo surface already resolves the canopy TOP well, so
anchoring it to GEDI canopy-top (noisy, 25 m footprint) can only match or harm. **Spaceborne
lidar's unique value is the UNDER-CANOPY GROUND, which optical cannot see.** That is M8.

## Feasibility (from reading the EOGS renderer)
The rendered altitude is NOT hard-coded in the CUDA kernel — it is passed as a channel of
`colors_precomp = cat([rgb, altitude, constant])`, with `altitude = xyz_uva[...,2]`. So a
SECOND **ground** altitude map can be rendered by a second rasterizer pass whose altitude
channel is a per-Gaussian ground height — **no CUDA recompile**. Validated by code inspection.

## The two implementation options

### Option A — GS-native two-surface (recommended end-state)
Add one learnable per-Gaussian parameter `g_i`; define ground height `a_ground_i = a_top_i -
softplus(g_i)` (>= 0 so ground is at/below the surface). Render `A_top` (as now, photometric +
appearance) and `A_ground` (2nd pass). Supervise:
- `A_top`  : photometric (unchanged).
- `A_ground` : GEDI **ground** anchors (Huber), projected at ground altitude.
- priors: `softplus(g)` small where no canopy (tree-mask) so ground=top on bare earth;
  TV smoothness on the ground field; ground <= top guaranteed by construction.
Cost: ~8 precise edits to the Gaussian model (optimizer group, densification clone/split,
prune, save/load) + a 2nd render pass + losses/args. Most faithful to "Gaussian Splatting",
view-consistent, strongest paper framing. Higher implementation risk.

### Option B — Lidar-supervised DTM field (fast, low-risk first result)
Keep EOGS untouched for the surface; add a separate learnable top-down **ground grid**
`Z_ground(x,y)` at the DSM resolution, supervised directly by GEDI ground anchors, regularized
to lie below the rendered optical surface and to equal it on non-canopy (tree-mask) pixels,
with TV smoothness. No Gaussian-model surgery; directly evaluable vs 3DEP bare-earth DTM.
Less "GS-native" (a bolt-on terrain head), but validates the scientific premise cheaply.

## Recommendation
Stage **B then A**: B gives a fast, low-risk REAL result answering the core question — does
spaceborne-lidar ground supervision recover under-canopy terrain better than using the optical
surface as the ground? If yes (premise validated), build A for the GS-native paper version.
This matches our rigor rule: validate the premise before the expensive elegant build.

## Evaluation (both options)
Render a top-down ground DTM; compare to USGS 3DEP bare-earth DTM (datum-aligned, validated in
scripts/13) on TREE pixels. Win metric: ground-DTM MAE under canopy < (optical-surface-as-
ground) MAE. Also confirm the canopy-top DSM MAE is unchanged (two-surface must not hurt the top).
