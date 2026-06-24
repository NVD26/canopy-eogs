#!/usr/bin/env python3
"""
apply_paper1_lidar.py — [Paper 1 / M7] idempotently patch an EOGS clone to add the
GEDI lidar canopy-anchor loss. Re-running is safe (each edit is guarded by a marker).

It (1) copies lidar_anchors.py into EOGS/src/gaussiansplatting/, then (2) inserts:
  - arguments/__init__.py : ModelParams.lidar_anchors_path, OptimizationParams.{w,iterstart}_L_lidar
  - train.py             : build the anchor index, init L_lidar, compute it from the
                           rendered altitude at the footprints, add it to the total loss.

Usage:  python eogs_mods/apply_paper1_lidar.py [EOGS_DIR]
        (EOGS_DIR defaults to $EOGS_DIR or ~/eogs-src/EOGS)
"""
import os, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
EOGS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EOGS_DIR", os.path.expanduser("~/eogs-src/EOGS"))
GS = os.path.join(EOGS, "src", "gaussiansplatting")
ARGS = os.path.join(GS, "arguments", "__init__.py")
TRAIN = os.path.join(GS, "train.py")

def fail(m): print("!! " + m); sys.exit(1)
for p in (GS, ARGS, TRAIN):
    if not os.path.exists(p): fail(f"not found: {p}  (is EOGS_DIR correct? got {EOGS})")

def patch(path, anchor, insert, marker, after=True, replace=False):
    """Insert `insert` relative to `anchor` unless `marker` already present.
       replace=True  -> replace `anchor` with `insert` (insert must contain anchor)."""
    s = open(path).read()
    if marker in s:
        print(f"   = already patched ({marker}) in {os.path.basename(path)}"); return s, False
    if anchor not in s:
        fail(f"anchor not found in {os.path.basename(path)}:\n   {anchor!r}")
    if replace:
        s2 = s.replace(anchor, insert, 1)
    elif after:
        s2 = s.replace(anchor, anchor + insert, 1)
    else:
        s2 = s.replace(anchor, insert + anchor, 1)
    open(path, "w").write(s2)
    print(f"   + patched ({marker}) in {os.path.basename(path)}")
    return s2, True

# 0) copy the anchor-index module next to train.py
shutil.copyfile(os.path.join(HERE, "lidar_anchors.py"), os.path.join(GS, "lidar_anchors.py"))
print(f"   + copied lidar_anchors.py -> {GS}")

# 1) ModelParams: a path to the anchors .npz
patch(ARGS,
      "        self.eval = False\n",
      "        self.lidar_anchors_path = \"\"  # [P1-lidar]\n",
      "lidar_anchors_path")

# 2) OptimizationParams: weight + start iter (defaults keep baseline behaviour: weight 0)
patch(ARGS,
      "        self.w_L_accumulated_opacity = 0.0\n",
      "        self.iterstart_L_lidar = -1            # [P1-lidar]\n"
      "        self.w_L_lidar = 0.0                   # [P1-lidar] GEDI canopy-anchor loss weight (0 = off)\n",
      "w_L_lidar = 0.0")

# 3) train.py: build per-view anchor index right after the Scene is created
patch(TRAIN,
      "    scene = Scene(dataset, gaussians)\n",
      "    # [P1-lidar] per-view GEDI canopy-top anchor index (empty unless --lidar_anchors_path given)\n"
      "    try:\n"
      "        from lidar_anchors import build_index as _p1_build_index\n"
      "        lidar_index = _p1_build_index(dataset.source_path, getattr(dataset, 'lidar_anchors_path', '') or '')\n"
      "        if lidar_index:\n"
      "            _p1n = sum(v['u'].numel() for v in lidar_index.values())\n"
      "            try:\n"
      "                _names = {c.image_name for c in scene.getTrainCameras()}\n"
      "            except Exception:\n"
      "                _names = set()\n"
      "            _hit = len(set(lidar_index) & _names) if _names else -1\n"
      "            print(f'[P1-lidar] {_p1n} anchor-views across {len(lidar_index)} cameras; {_hit} match train-cams from ' + dataset.lidar_anchors_path)\n"
      "            if _names and _hit == 0:\n"
      "                print('[P1-lidar] !! WARNING: anchor names match no train camera -> ZERO supervision. Check naming.')\n"
      "    except Exception as _e:\n"
      "        print('[P1-lidar] anchor load skipped:', _e); lidar_index = {}\n",
      "_p1_build_index")

# 4) train.py: initialise L_lidar each iteration, next to the other loss-term inits
patch(TRAIN,
      "        L_accumulated_opacity = 0\n",
      "        L_lidar = 0  # [P1-lidar]\n",
      "L_lidar = 0  # [P1-lidar]")

# 5) train.py: compute L_lidar from the rendered altitude at the footprint pixels,
#    just before the total loss is assembled.
patch(TRAIN,
      "        loss = (\n",
      "        if lidar_index and iteration > opt.iterstart_L_lidar:  # [P1-lidar] compute\n"
      "            _anc = lidar_index.get(viewpoint_cam.image_name)\n"
      "            if _anc is not None and _anc['u'].numel() > 0:\n"
      "                _grid = torch.stack([_anc['u'], _anc['v']], dim=-1).view(1, -1, 1, 2)\n"
      "                _h, _w = altitude_render.shape\n"
      "                _samp = torch.nn.functional.grid_sample(\n"
      "                    altitude_render.view(1, 1, _h, _w), _grid, align_corners=True).view(-1)\n"
      "                L_lidar = (_samp - _anc['canopytop']).abs().mean()\n",
      "[P1-lidar] compute", after=False)

# 6) train.py: add the term to the total loss
patch(TRAIN,
      "            + opt.w_L_accumulated_opacity * L_accumulated_opacity\n",
      "            + opt.w_L_lidar * L_lidar  # [P1-lidar]\n",
      "+ opt.w_L_lidar * L_lidar")

print("DONE. EOGS patched for the Paper-1 lidar loss. Re-running this is safe (idempotent).")
