"""
lidar_anchors.py — [Paper 1] per-view GEDI anchor index for EOGS lidar losses.
build_index(..., target) projects each footprint at the chosen height ('canopytop' or 'ground')
— parallax depends on altitude — and returns the NDC (u,v) where it lands plus the target height
(metres, WGS84 ellipsoid = EOGS frame). Verified vs the satellite RPC by scripts/14 (~0.3 px).
"""
import os, json
import numpy as np
try:
    import torch; _T = True
except ImportError:
    _T = False

def build_index(source_path, anchors_npz, target="canopytop", device="cuda"):
    if not anchors_npz or not os.path.exists(anchors_npz):
        return {}
    aff = os.path.join(source_path, "affine_models.json")
    if not os.path.exists(aff):
        print(f"[P1] no affine_models.json at {source_path}"); return {}
    metas = json.load(open(aff))
    a = np.load(anchors_npz, allow_pickle=True)
    E, N = a["E"].astype("float64"), a["N"].astype("float64")
    top = a["canopytop"].astype("float64"); gnd = a["ground"].astype("float64")
    proj_h = gnd if target == "ground" else top
    center = np.array(metas[0]["model"]["center"], dtype="float64")
    scale = float(metas[0]["model"]["scale"])
    world = (np.stack([E, N, proj_h], axis=-1) - center) / scale
    idx = {}
    for m in metas:
        if m.get("img") == "Nadir":
            continue
        A = np.array(m["model"]["coef_"], dtype="float64")
        b = np.array(m["model"]["intercept_"], dtype="float64")
        uva = world @ A.T + b
        u, v = uva[:, 0], uva[:, 1]
        inv = (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
        if not inv.any():
            continue
        name = m["img"].replace(".tif", "")
        if _T:
            t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device)
            idx[name] = {"u": t(u[inv]), "v": t(v[inv]), "h": t(proj_h[inv]),
                         "canopytop": t(top[inv]), "ground": t(gnd[inv])}
        else:
            idx[name] = {"u": u[inv], "v": v[inv], "h": proj_h[inv],
                         "canopytop": top[inv], "ground": gnd[inv]}
    return idx
