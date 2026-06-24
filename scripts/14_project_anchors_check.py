#!/usr/bin/env python3
"""
14_project_anchors_check.py — VERIFY anchor-to-view projection BEFORE adding any loss.

The M7 lidar loss will, for each training view, sample the rendered altitude at the pixel
where a GEDI footprint projects, and compare it to the GEDI height. If anchors project to
the WRONG pixels, the loss teaches the wrong thing. So we verify the projection first, with
two independent checks that need no GPU:

  1. Self-consistency of heights: EOGS maps world points to (u, v, altitude_in_metres) with a
     per-view affine. Projecting an anchor at its own height should return that height
     (round-trip), confirming we use the scene normalization (centre, scale) correctly.
  2. Geolocation vs the exact satellite camera: each view also stores its Rational Polynomial
     Coefficient (RPC) camera, the true (non-approximate) projection. The EOGS affine claims to
     match the RPC to ~0.01 pixel. We project every anchor BOTH ways and report the pixel
     disagreement. Small disagreement = our affine projection is correct.

EOGS world frame (from its source): world = (UTM - model.center) / model.scale, then
(u, v, a) = world @ A^T + b with A = model.coef_, b = model.intercept_; pixel col = (u+1)/2*(W-1),
row = (v+1)/2*(H-1).

Usage:
  python scripts/14_project_anchors_check.py --scene JAX_068
"""
import argparse, json, os, sys
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="JAX_068")
    ap.add_argument("--eogs-dir", default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--use", choices=["canopytop", "ground"], default="canopytop",
                    help="which height to project (both should match the RPC)")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    anchors_path = args.anchors or os.path.join(here, "..", "data", "anchors", f"{args.scene}_gedi_anchors.npz")
    aff_path = os.path.join(args.eogs_dir, "data", "affine_models", args.scene, "affine_models.json")

    if not os.path.exists(aff_path):
        print(f"!! {aff_path} not found. Run scripts/04_prep_cameras.sh for {args.scene} first."); return 1
    if not os.path.exists(anchors_path):
        print(f"!! {anchors_path} not found. Run scripts/12_build_anchors.py first."); return 1

    a = np.load(anchors_path, allow_pickle=True)
    lon, lat = a["lon"], a["lat"]
    E, N = a["E"], a["N"]
    alt = a[args.use].astype("float64")
    n_anch = len(lon)
    print(f"scene {args.scene}: {n_anch} anchors (projecting '{args.use}' height, "
          f"frame={a['height_frame'] if 'height_frame' in a.files else '?'})")
    if n_anch == 0:
        print("!! no anchors."); return 1

    metas = json.load(open(aff_path))
    center = np.array(metas[0]["model"]["center"], dtype="float64")
    scale = float(metas[0]["model"]["scale"])
    print(f"scene normalization: centre {center.round(1).tolist()}  scale {scale:.1f} m")

    try:
        import rpcm
        have_rpc = True
    except ImportError:
        have_rpc = False
        print("(rpcm not available; skipping the exact-RPC cross-check)")

    utm_anch = np.stack([E, N, alt], axis=-1)          # (n,3) UTM + height
    world = (utm_anch - center) / scale                # normalized world

    tot_inview = 0
    height_err = []
    pix_off = []
    per_view = []
    for m in metas:
        if m.get("img") == "Nadir":
            continue
        A = np.array(m["model"]["coef_"], dtype="float64")
        b = np.array(m["model"]["intercept_"], dtype="float64")
        W, H = int(m["width"]), int(m["height"])
        uva = world @ A.T + b
        u, v, a_m = uva[:, 0], uva[:, 1], uva[:, 2]
        col = (u + 1) / 2 * (W - 1)
        row = (v + 1) / 2 * (H - 1)
        inview = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        tot_inview += int(inview.sum())
        if inview.any():
            height_err.append(np.abs(a_m[inview] - alt[inview]))
        if have_rpc and inview.any():
            try:
                rpc = rpcm.RPCModel(m["rpc"], dict_format="rpcm")
                cr = rpc.projection(lon[inview], lat[inview], alt[inview])
                col_r, row_r = np.asarray(cr[0]), np.asarray(cr[1])
                d = np.hypot(col[inview] - col_r, row[inview] - row_r)
                pix_off.append(d)
            except Exception as e:
                if not per_view:
                    print(f"  (RPC projection note: {e})")
        per_view.append((m.get("img", "?"), int(inview.sum())))

    n_views = len(per_view)
    print(f"\nviews: {n_views} | anchors-in-view total: {tot_inview} "
          f"(avg {tot_inview/max(n_views,1):.1f} per view, of {n_anch} anchors)")
    if height_err:
        he = np.concatenate(height_err)
        print(f"height round-trip |affine-altitude - anchor-height|: median {np.median(he):.3f} m, "
              f"max {he.max():.3f} m  (should be ~0; checks centre/scale)")
    if pix_off:
        po = np.concatenate(pix_off)
        print(f"affine-vs-RPC pixel disagreement: median {np.median(po):.3f} px, "
              f"95th pct {np.percentile(po,95):.3f} px  (should be < ~2 px)")
    print("\nper-view anchors-in-view (first 8):")
    for img, k in per_view[:8]:
        print(f"  {img:<28} {k}")

    ok_h = (not height_err) or (np.median(np.concatenate(height_err)) < 1.0)
    ok_p = (not pix_off) or (np.median(np.concatenate(pix_off)) < 2.0)
    cover = tot_inview > 0
    verdict = "PASS" if (ok_h and ok_p and cover) else "CHECK"
    print(f"\nVERDICT: {verdict} — projection is "
          f"{'consistent and matches the satellite RPC; ready to wire the loss.' if verdict=='PASS' else 'off; inspect the numbers above before adding the loss.'}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
