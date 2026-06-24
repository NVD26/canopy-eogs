#!/usr/bin/env python3
"""
apply_paper1_twosurface.py — [Paper 1 / M8] idempotently patch an EOGS clone to add the
GS-NATIVE two-surface model: each Gaussian gets a ground height a_ground = a_top - softplus(g);
a second render pass produces a ground altitude map A_ground, supervised by GEDI GROUND anchors
(+ a collapse prior so ground=surface unless lidar pulls it down). No CUDA recompile.

Apply to a FRESH EOGS clone (do not combine with apply_paper1_lidar.py).
Usage: python eogs_mods/apply_paper1_twosurface.py [EOGS_DIR]
"""
import os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
EOGS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EOGS_DIR", os.path.expanduser("~/eogs-src/EOGS"))
GS = os.path.join(EOGS, "src", "gaussiansplatting")
ARGS = os.path.join(GS, "arguments", "__init__.py")
TRAIN = os.path.join(GS, "train.py")
GM = os.path.join(GS, "scene", "gaussian_model.py")
RENDER = os.path.join(GS, "gaussian_renderer", "__init__.py")
RENDERPY = os.path.join(GS, "render.py")
for p in (GS, ARGS, TRAIN, GM, RENDER, RENDERPY):
    if not os.path.exists(p):
        print("!! not found:", p); sys.exit(1)

def patch(path, anchor, insert, marker, mode="after"):
    s = open(path).read()
    if marker in s:
        print(f"   = already ({marker[:34]}) in {os.path.basename(path)}"); return
    if anchor not in s:
        print(f"!! anchor missing in {os.path.basename(path)}: {anchor[:60]!r}"); sys.exit(1)
    if mode == "replace":
        s = s.replace(anchor, insert, 1)
    elif mode == "after":
        s = s.replace(anchor, anchor + insert, 1)
    else:
        s = s.replace(anchor, insert + anchor, 1)
    open(path, "w").write(s); print(f"   + patched ({marker[:34]}) in {os.path.basename(path)}")

shutil.copyfile(os.path.join(HERE, "lidar_anchors.py"), os.path.join(GS, "lidar_anchors.py"))
print("   + copied lidar_anchors.py")

# ---- gaussian_renderer: allow a geometry-detached pass (for the ground surface) ----
patch(RENDER,
      "def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):",
      "def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, detach_geometry = False):",
      'detach_geometry = False', mode="replace")
patch(RENDER, "    if override_color is None:\n",
      "    if detach_geometry:  # [P1-2surf] ground pass: freeze geometry, train only the ground offset\n"
      "        means3D = means3D.detach(); opacity = opacity.detach()\n"
      "        scales = scales.detach() if scales is not None else None\n"
      "        rotations = rotations.detach() if rotations is not None else None\n"
      "        cov3D_precomp = cov3D_precomp.detach() if cov3D_precomp is not None else None\n",
      'if detach_geometry:', mode="before")

# ---- gaussian_model.py: per-Gaussian ground parameter plumbing ----
patch(GM, "        self._opacity = torch.empty(0)\n",
      "        self._ground = torch.empty(0)\n", 'self._ground = torch.empty(0)')
patch(GM, "        self._opacity = nn.Parameter(opacities.requires_grad_(True))\n",
      "        self._ground = nn.Parameter((-10.0 * torch.ones((fused_point_cloud.shape[0], 1), device=\"cuda\")).requires_grad_(True))  # [P1-2surf]\n",
      'self._ground = nn.Parameter')
patch(GM,
      "            {'params': [self._rotation], 'lr': training_args.rotation_lr, \"name\": \"rotation\"}\n        ]",
      "            {'params': [self._rotation], 'lr': training_args.rotation_lr, \"name\": \"rotation\"},\n"
      "            {'params': [self._ground], 'lr': getattr(training_args, 'ground_lr', 0.01), \"name\": \"ground\"}  # [P1-2surf]\n        ]",
      '"name": "ground"', mode="replace")
patch(GM,
      "def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):",
      "def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ground=None):",
      'new_ground=None', mode="replace")
patch(GM, "        \"rotation\" : new_rotation}",
      "        \"rotation\" : new_rotation,\n        \"ground\": new_ground}", '"ground": new_ground', mode="replace")
patch(GM,
      "        self._rotation = optimizable_tensors[\"rotation\"]\n\n        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")",
      "        self._rotation = optimizable_tensors[\"rotation\"]\n        self._ground = optimizable_tensors[\"ground\"]  # [P1-2surf]\n\n        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=\"cuda\")",
      'optimizable_tensors["ground"]  # [P1-2surf]', mode="replace")
patch(GM,
      "        self._rotation = optimizable_tensors[\"rotation\"]\n\n        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]",
      "        self._rotation = optimizable_tensors[\"rotation\"]\n        self._ground = optimizable_tensors[\"ground\"]\n\n        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]",
      'optimizable_tensors["ground"]\n\n        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]', mode="replace")
patch(GM, "        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)\n",
      "        new_ground = self._ground[selected_pts_mask].repeat(N,1)  # [P1-2surf]\n",
      'new_ground = self._ground[selected_pts_mask].repeat(N,1)')
patch(GM,
      "self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)",
      "self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_ground=new_ground)",
      'new_opacity, new_scaling, new_rotation, new_ground=new_ground', mode="replace")
patch(GM,
      "        new_rotation = self._rotation[selected_pts_mask]\n\n        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)",
      "        new_rotation = self._rotation[selected_pts_mask]\n        new_ground = self._ground[selected_pts_mask]\n\n        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ground=new_ground)",
      'new_opacities, new_scaling, new_rotation, new_ground=new_ground', mode="replace")

# ---- arguments/__init__.py ----
patch(ARGS, "        self.eval = False\n",
      "        self.lidar_anchors_path = \"\"  # [P1-2surf]\n", 'lidar_anchors_path')
patch(ARGS, "        self.w_L_accumulated_opacity = 0.0\n",
      "        self.iterstart_L_ground = -1           # [P1-2surf]\n"
      "        self.w_L_ground = 0.0                  # [P1-2surf] GEDI ground supervision weight (0=off)\n"
      "        self.w_L_groundcollapse = 0.01         # [P1-2surf] prior: ground=surface unless lidar pulls down\n"
      "        self.ground_lr = 0.01                  # [P1-2surf]\n"
      "        self.w_L_groundtv = 0.1                 # [P1-2surf] render-space ground smoothness\n", 'w_L_ground = 0.0')

# ---- train.py ----
patch(TRAIN, "    scene = Scene(dataset, gaussians)\n",
      "    # [P1-2surf] per-view GEDI GROUND anchor index\n"
      "    try:\n"
      "        from lidar_anchors import build_index as _p1_idx\n"
      "        ground_index = _p1_idx(dataset.source_path, getattr(dataset, 'lidar_anchors_path', '') or '', target='ground')\n"
      "        if ground_index:\n"
      "            _gn = sum(v['u'].numel() for v in ground_index.values())\n"
      "            print(f'[P1-2surf] {_gn} ground anchor-views across {len(ground_index)} cams from ' + dataset.lidar_anchors_path)\n"
      "    except Exception as _e:\n"
      "        print('[P1-2surf] ground anchor load skipped:', _e); ground_index = {}\n",
      '_p1_idx')
patch(TRAIN, "        L_accumulated_opacity = 0\n",
      "        L_ground = 0  # [P1-2surf]\n        L_groundcollapse = 0  # [P1-2surf]\n        L_groundtv = 0  # [P1-2surf]\n",
      'L_ground = 0  # [P1-2surf]')
patch(TRAIN, "        loss = (\n",
      "        if ground_index and iteration > opt.iterstart_L_ground:  # [P1-2surf] ground render+loss\n"
      "            _Np1 = gaussians._xyz.shape[0]\n"
      "            _uva = viewpoint_cam.ECEF_to_UVA(gaussians._xyz)\n"
      "            _ag = _uva[..., 2:3].detach() - torch.nn.functional.softplus(gaussians._ground)  # detach top: ground loss trains only the offset\n"
      "            _gc = torch.cat([torch.zeros(_Np1, 3, device=_ag.device), _ag, torch.ones(_Np1, 1, device=_ag.device)], dim=-1)\n"
      "            _Ag = render(viewpoint_cam, gaussians, pipe, bg, override_color=_gc, detach_geometry=True)[\"render\"][3]\n"
      "            L_groundtv = (_Ag[1:, :] - _Ag[:-1, :]).abs().mean() + (_Ag[:, 1:] - _Ag[:, :-1]).abs().mean()  # [P1-2surf] smoothness\n"
      "            L_groundcollapse = torch.nn.functional.softplus(gaussians._ground).mean()\n"
      "            _anc = ground_index.get(viewpoint_cam.image_name)\n"
      "            if _anc is not None and _anc['u'].numel() > 0:\n"
      "                _grid = torch.stack([_anc['u'], _anc['v']], dim=-1).view(1, -1, 1, 2)\n"
      "                _h, _w = _Ag.shape\n"
      "                _s = torch.nn.functional.grid_sample(_Ag.view(1, 1, _h, _w), _grid, align_corners=True).view(-1)\n"
      "                L_ground = torch.nn.functional.huber_loss(_s, _anc['h'], delta=2.0)\n"
      "                if iteration % 500 == 0:\n"
      "                    _stop = torch.nn.functional.grid_sample(altitude_render.view(1,1,_h,_w), _grid, align_corners=True).view(-1)\n"
      "                    print(f\"[P1-2surf] it{iteration} |ground-GEDI|={(_s-_anc['h']).abs().median().item():.2f}m |top-GEDI|={(_stop-_anc['h']).abs().median().item():.2f}m n={_s.numel()}\")\n",
      '[P1-2surf] ground render+loss', mode="before")
patch(TRAIN, "            + opt.w_L_accumulated_opacity * L_accumulated_opacity\n",
      "            + opt.w_L_ground * L_ground  # [P1-2surf]\n            + opt.w_L_groundcollapse * L_groundcollapse\n            + opt.w_L_groundtv * L_groundtv\n",
      'opt.w_L_ground * L_ground')

# train.py: dump the two-surface state after the Gaussians are saved (for M9 ground eval)
patch(TRAIN, "                scene.save(iteration)\n",
      "                import numpy as _np2  # [P1-2surf] dump ground state for evaluation\n"
      "                _np2.savez(scene.model_path + \"/twosurf_state.npz\",\n"
      "                           xyz=gaussians._xyz.detach().cpu().numpy(),\n"
      "                           ground=gaussians._ground.detach().cpu().numpy(),\n"
      "                           opacity=gaussians.get_opacity.detach().cpu().numpy())\n",
      'twosurf_state.npz')

# corrective A: render() must accept detach_geometry
_r = open(RENDER).read()
if "detach_geometry = False" not in _r and "scaling_modifier = 1.0, override_color = None):" in _r:
    _r = _r.replace("scaling_modifier = 1.0, override_color = None):",
                    "scaling_modifier = 1.0, override_color = None, detach_geometry = False):")
    if "if detach_geometry:" not in _r:
        _r = _r.replace("    if override_color is None:\n",
            "    if detach_geometry:\n        means3D = means3D.detach(); opacity = opacity.detach()\n"
            "        scales = scales.detach() if scales is not None else None\n"
            "        rotations = rotations.detach() if rotations is not None else None\n"
            "        cov3D_precomp = cov3D_precomp.detach() if cov3D_precomp is not None else None\n"
            "    if override_color is None:\n", 1)
    open(RENDER, "w").write(_r); print("   + corrected: render() detach_geometry flag added")
# corrective B: ground render call must request detach_geometry
_t0 = open(TRAIN).read()
if 'override_color=_gc)["render"][3]' in _t0:
    _t0 = _t0.replace('override_color=_gc)["render"][3]', 'override_color=_gc, detach_geometry=True)["render"][3]')
    open(TRAIN, "w").write(_t0); print("   + corrected: ground render now detaches geometry")
# corrective: ensure the ground render detaches the top altitude (fixes earlier non-detached patch)
_t = open(TRAIN).read()
if "_uva[..., 2:3] - torch.nn.functional.softplus(gaussians._ground)" in _t:
    _t = _t.replace("_uva[..., 2:3] - torch.nn.functional.softplus(gaussians._ground)",
                    "_uva[..., 2:3].detach() - torch.nn.functional.softplus(gaussians._ground)")
    open(TRAIN, "w").write(_t); print("   + corrected: detached top altitude in ground render")

# ---- render.py: emit a faithful (alpha-composite) GROUND DSM, loading the dumped offset ----
patch(RENDERPY, "        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)\n",
      "        import os as _os2, numpy as _np3, torch as _t3  # [P1-2surf] load learned ground offset\n"
      "        _tp = _os2.path.join(dataset.model_path, \"twosurf_state.npz\")\n"
      "        if _os2.path.exists(_tp):\n"
      "            gaussians._ground = _t3.tensor(_np3.load(_tp)[\"ground\"], dtype=_t3.float, device=\"cuda\")\n"
      "            print(\"[P1-2surf] loaded ground offset\", tuple(gaussians._ground.shape))\n",
      '[P1-2surf] load learned ground')
patch(RENDERPY, "        cloud = view.UVA_to_ECEF(rendered_uva.detach().reshape((-1, 3))).cpu().numpy()\n",
      "        _cloud_g = None  # [P1-2surf] ground surface render\n"
      "        if getattr(gaussians, \"_ground\", torch.empty(0)).numel() == gaussians._xyz.shape[0]:\n"
      "            _uvaG = view.ECEF_to_UVA(gaussians._xyz)\n"
      "            _agc = _uvaG[..., 2:3].detach() - torch.nn.functional.softplus(gaussians._ground)\n"
      "            _Ng = _agc.shape[0]\n"
      "            _gcc = torch.cat([torch.zeros(_Ng, 3, device=_agc.device), _agc, torch.ones(_Ng, 1, device=_agc.device)], dim=-1)\n"
      "            _Agr = render(view, gaussians, pipeline, background, override_color=_gcc, detach_geometry=True)[\"render\"][3]\n"
      "            _uva_g = torch.stack(view.UV_grid + (_Agr,), dim=-1)\n"
      "            _cloud_g = view.UVA_to_ECEF(_uva_g.detach().reshape((-1, 3))).cpu().numpy()\n",
      '[P1-2surf] ground surface render')
patch(RENDERPY,
      "        with rasterio.open(os.path.join(base_path, \"dsm\", name), \"w\", **profile) as f:\n            f.write(dsm[:, :, 0], 1)\n",
      "        if _cloud_g is not None:  # [P1-2surf] write ground DSM on the SAME grid as the top DSM\n"
      "            _cloud_g = _cloud_g * scene_params[1] + scene_params[0]\n"
      "            _dsmg = plyflatten(_cloud_g, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float(\"inf\"))\n"
      "            os.makedirs(os.path.join(base_path, \"ground_dsm\"), exist_ok=True)\n"
      "            with rasterio.open(os.path.join(base_path, \"ground_dsm\", name), \"w\", **profile) as f:\n"
      "                f.write(_dsmg[:, :, 0], 1)\n",
      '[P1-2surf] write ground DSM')

# corrective: TV smoothness (existing clones)
_t1 = open(TRAIN).read(); _ch=False
if "L_groundtv = (" not in _t1:
    _t1 = _t1.replace('override_color=_gc, detach_geometry=True)["render"][3]\n',
        'override_color=_gc, detach_geometry=True)["render"][3]\n'
        '            L_groundtv = (_Ag[1:, :] - _Ag[:-1, :]).abs().mean() + (_Ag[:, 1:] - _Ag[:, :-1]).abs().mean()\n', 1); _ch=True
if "L_groundtv = 0  # [P1-2surf]" not in _t1:
    _t1 = _t1.replace("        L_groundcollapse = 0  # [P1-2surf]\n",
                      "        L_groundcollapse = 0  # [P1-2surf]\n        L_groundtv = 0  # [P1-2surf]\n", 1); _ch=True
if "opt.w_L_groundtv * L_groundtv" not in _t1:
    _t1 = _t1.replace("            + opt.w_L_groundcollapse * L_groundcollapse\n",
                      "            + opt.w_L_groundcollapse * L_groundcollapse\n            + opt.w_L_groundtv * L_groundtv\n", 1); _ch=True
if _ch: open(TRAIN,"w").write(_t1); print("   + corrected: ground TV smoothness wired in train.py")
_ar = open(ARGS).read()
if "w_L_groundtv" not in _ar:
    _ar = _ar.replace("        self.ground_lr = 0.01                  # [P1-2surf]\n",
        "        self.ground_lr = 0.01                  # [P1-2surf]\n        self.w_L_groundtv = 0.1                 # [P1-2surf] render-space ground smoothness\n", 1)
    open(ARGS,"w").write(_ar); print("   + corrected: w_L_groundtv arg added")

print("DONE. EOGS patched for the Paper-1 two-surface model (idempotent).")
