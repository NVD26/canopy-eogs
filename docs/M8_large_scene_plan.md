# M8 larger-scene build plan (Paper 1)

Goal: an EOGS scene over a ~1 km Jacksonville AOI with ~166 validated GEDI ground anchors, so
the lidar GROUND supervision (M8) can be trained and evaluated against 3DEP bare-earth DTM.

Chosen AOI: centre UTM (434910, 3357367) zone 17N / lon-lat (-81.6772, 30.3464), ~1 km square.

## Lineage (why this route)
EOGS consumes Sat-NeRF-style scenes: per-image RGB crop + RPC + alt bounds + sun angles, then
`to_affine.py`. Sat-NeRF's `create_satellite_dataset.py` builds exactly this from the open
DFC2019 Track3-RGB + Track3-Truth, applying BUNDLE ADJUSTMENT (relative RPC correction) which
EOGS depends on (raw RPCs disagree by tens of pixels across views). So we reuse that pipeline
for a larger AOI instead of reinventing photogrammetry.

## Stages (each verified before the next)
1. TOOLING (scripts/19_setup_scene_tools.sh): clone centreborelli/satnerf + sat-bundleadjust;
   create the `ba` conda env (bundle_adjust + s2p deps). Fold any install fixes back into the
   script (advisor reproducibility rule).
2. DATA: download DFC2019 Track3-RGB + Track3-Truth for Jacksonville (open access, IEEE
   DataPort / pubgeo dfc2019). ~tens of GB; needs disk + possibly a DataPort login.
3. SCENE: run create_satellite_dataset for our custom ~1 km AOI (extend it to accept a bbox /
   merged truth, with --ba). Output EOGS-format images + metadata; run to_affine.
4. BASELINE: train EOGS on the new scene; eval surface MAE vs airborne DSM (expect ~1-2 m) to
   confirm the larger scene reconstructs well before adding lidar.
5. M8-B: run scripts/17 on the new scene (now ~166 anchors) -> first real ground-DTM result
   vs 3DEP (lidar vs priors-only ablation).
6. M8-A: build the GS-native two-surface (per-Gaussian ground + 2nd render pass) on the scene;
   compare to M8-B. This is the paper's method.

## Evaluation
Under-canopy ground MAE vs 3DEP bare-earth on tall-canopy pixels; baseline = optical surface as
ground (~15 m error, measured). Also confirm canopy-top DSM MAE is unchanged.

## Risks
- s2p / bundle_adjust C-dependency install (GDAL, etc.) — iterate like we did for EOGS.
- DFC2019 source size/login — may need manual download step.
- Custom AOI vs per-tile truth — create script is tile-centric; we extend it to our bbox.
