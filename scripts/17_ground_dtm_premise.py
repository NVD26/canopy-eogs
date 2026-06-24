#!/usr/bin/env python3
"""
17_ground_dtm_premise.py — [Paper 1 / M8-B] validate the under-canopy premise on REAL data.

Premise: EOGS's optical surface reconstructs the CANOPY TOP well but sits well ABOVE the true
ground under trees; sparse GEDI GROUND returns + regularization can pull a ground surface down
toward the true terrain. We test this WITHOUT touching the Gaussian model: a learnable top-down
ground grid Z is fit on the eval grid from (a) GEDI ground anchors, (b) the EOGS surface as a
ceiling, (c) ground=surface on non-canopy pixels, (d) TV smoothness. Truth = USGS 3DEP DTM.

All heights are put in the GT/3DEP datum (NAVD88): EOGS RPC heights (and the DFC2019 GT DSM
registered to them) are ELLIPSOIDAL -> converted via the npz geoid offset; GEDI ground likewise.

Usage (on the 4090, env 'eogs'):
  python scripts/17_ground_dtm_premise.py --scene JAX_068 --rdsm <...>/JAX_068_rdsm.tif --dry-run
  python scripts/17_ground_dtm_premise.py --scene JAX_068 --rdsm <...>/JAX_068_rdsm.tif
"""
import os, sys, argparse, json, urllib.request, urllib.parse
import numpy as np

def log(*a): print(*a, flush=True)

USGS = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevation/ImageServer/exportImage")
def fetch_3dep_dtm(w, s, e, n, npix_w, npix_h, out_tif):
    params = {"bbox": f"{w},{s},{e},{n}", "bboxSR": "4326", "imageSR": "4326",
              "size": f"{npix_w},{npix_h}", "format": "tiff",
              "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    data = urllib.request.urlopen(USGS + "?" + urllib.parse.urlencode(params), timeout=180).read()
    if data[:2] not in (b"II", b"MM"):
        raise RuntimeError(f"3DEP did not return a GeoTIFF ({len(data)} bytes)")
    with open(out_tif, "wb") as f:
        f.write(data)
    return out_tif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="JAX_068")
    ap.add_argument("--eogs-dir", default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--rdsm", required=True, help="registered EOGS DSM (<scene>_rdsm.tif from an eval run)")
    ap.add_argument("--anchors", default=None, help="GEDI anchors npz (default repo data/anchors/<scene>_gedi_anchors.npz)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--w-lidar", type=float, default=1.0)
    ap.add_argument("--w-ceil", type=float, default=0.5)
    ap.add_argument("--w-bare", type=float, default=1.0)
    ap.add_argument("--w-tv", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling, transform_bounds

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = args.scene
    gt_dir = os.path.join(args.eogs_dir, "data", "truth", scene)
    gt_dsm_path = os.path.join(gt_dir, f"{scene}_DSM.tif")
    gt_txt = os.path.join(gt_dir, f"{scene}_DSM.txt")
    tree_path = os.path.join(args.eogs_dir, "scripts", "eval", "tree_masks", f"{scene}.png")
    anchors = args.anchors or os.path.join(repo, "data", "anchors", f"{scene}_gedi_anchors.npz")
    out = args.out or os.path.join(repo, "results", f"ground_dtm_{scene}")
    os.makedirs(out, exist_ok=True)

    for p in (gt_dsm_path, gt_txt, args.rdsm, anchors):
        if not os.path.exists(p):
            log(f"!! missing input: {p}"); return 2
    if not os.path.exists(tree_path):
        log(f"!! missing tree mask: {tree_path}"); return 2

    meta = np.loadtxt(gt_txt)
    xoff, yoff_bottom, size, res = float(meta[0]), float(meta[1]), int(meta[2]), float(meta[3])
    ytop = yoff_bottom + size * res
    transform = from_origin(xoff, ytop, res, res)
    H = W = size
    log(f"[grid] {scene}: {W}x{H} @ {res} m  UTM xoff={xoff} yoff_bottom={yoff_bottom} ytop={ytop}")

    # GT DSM .tif has no CRS (plain raster); UTM zone + geoid from the anchors npz.
    a = np.load(anchors, allow_pickle=True)
    utm_crs = rasterio.crs.CRS.from_user_input(str(a["utm_epsg"]))
    geoid = float(a["geoid_offset_to_navd88"])   # ellipsoidal - NAVD88 (~ -29.84 near JAX)
    # EOGS RPC heights (and the GT DSM registered to them) are ELLIPSOIDAL; 3DEP + the GEDI
    # anchors we use are NAVD88. Put every surface in NAVD88: navd = ellip - geoid.
    with rasterio.open(gt_dsm_path) as f:
        gt_dsm = f.read(1).astype(np.float64)
    if gt_dsm.shape != (H, W):
        g = np.full((H, W), np.nan); hh = min(H, gt_dsm.shape[0]); ww = min(W, gt_dsm.shape[1])
        g[:hh, :ww] = gt_dsm[:hh, :ww]; gt_dsm = g
    gt_dsm = gt_dsm - geoid
    log(f"[grid] CRS = {utm_crs}; geoid {geoid:+.2f} m applied to surfaces (-> NAVD88)")

    with rasterio.open(args.rdsm) as f:
        rdsm = f.read(1).astype(np.float64)
    if rdsm.shape != (H, W):
        r = np.full((H, W), np.nan); h = min(H, rdsm.shape[0]); w = min(W, rdsm.shape[1])
        r[:h, :w] = rdsm[:h, :w]; rdsm = r
    rdsm[~np.isfinite(rdsm)] = np.nan
    rdsm = rdsm - geoid

    w_lon, s_lat, e_lon, n_lat = transform_bounds(utm_crs, "EPSG:4326",
                                                  xoff, yoff_bottom, xoff + W * res, ytop)
    tmptif = os.path.join(out, "_3dep_4326.tif")
    fetch_3dep_dtm(w_lon, s_lat, e_lon, n_lat, W, H, tmptif)
    dtm3dep = np.full((H, W), np.nan, dtype=np.float64)
    with rasterio.open(tmptif) as src:
        reproject(source=rasterio.band(src, 1), destination=dtm3dep,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=utm_crs, resampling=Resampling.bilinear)
    dtm3dep[dtm3dep < -1e3] = np.nan

    with rasterio.open(tree_path) as f:
        tm = f.read(1)
    tree = tm > 0.5
    if tree.shape != (H, W):
        t = np.zeros((H, W), bool); h = min(H, tree.shape[0]); w = min(W, tree.shape[1])
        t[:h, :w] = tree[:h, :w]; tree = t

    E, N = a["E"].astype(np.float64), a["N"].astype(np.float64)
    g_navd = a["ground"].astype(np.float64) - geoid
    col = np.round((E - xoff) / res).astype(int)
    row = np.round((ytop - N) / res).astype(int)
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    arow, acol, aval = row[inside], col[inside], g_navd[inside]
    n_anch = int(inside.sum())
    log(f"[anchors] {n_anch} GEDI ground anchors inside the {W*res:.0f} m tile (of {len(E)} in AOI)")

    treevalid = tree & np.isfinite(rdsm) & np.isfinite(dtm3dep) & np.isfinite(gt_dsm)
    def mae(x, y, m): d = (x - y)[m]; return float(np.nanmean(np.abs(d))) if np.size(d) else float("nan")
    def med(x, y, m): d = (x - y)[m]; return float(np.nanmedian(d)) if np.size(d) else float("nan")
    log("\n================ SANITY (must look right BEFORE fitting) ================")
    log(f"  tree pixels (valid): {int(treevalid.sum())} of {H*W}")
    log(f"  EOGS surface vs GT airborne DSM (tree)   MAE = {mae(rdsm, gt_dsm, treevalid):.2f} m   (expect ~1 m)")
    log(f"  GT airborne DSM vs 3DEP bare-earth (tree) median = {med(gt_dsm, dtm3dep, treevalid):+.2f} m   (= canopy height; expect +)")
    log(f"  EOGS surface vs 3DEP bare-earth (tree)    MAE = {mae(rdsm, dtm3dep, treevalid):.2f} m   (THE PROBLEM)")
    canopy_h = gt_dsm - dtm3dep
    tall = treevalid & np.isfinite(canopy_h) & (canopy_h > 3.0)
    log(f"  tall-canopy pixels (airborne canopy > 3 m): {int(tall.sum())}  (median height {np.nanmedian(canopy_h[tall]):+.1f} m)")
    log(f"  EOGS surface vs 3DEP on TALL canopy        MAE = {mae(rdsm, dtm3dep, tall):.2f} m   (under-canopy headroom)")
    if n_anch:
        cell3dep = dtm3dep[arow, acol]; ok = np.isfinite(cell3dep)
        log(f"  GEDI ground (NAVD88) vs 3DEP at anchor cells: median {np.nanmedian((aval-cell3dep)[ok]):+.2f} m, "
            f"abs-median {np.nanmedian(np.abs((aval-cell3dep)[ok])):.2f} m  (expect ~0; datum+placement)")
    log("========================================================================\n")

    if args.dry_run:
        log("DRY-RUN: inputs loaded and aligned. Re-run without --dry-run to fit the ground DTM.")
        return 0
    if n_anch == 0:
        log("!! no GEDI ground anchors inside this tile — cannot fit here (use a larger scene).")
        return 1

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rdsm_t = torch.tensor(np.nan_to_num(rdsm, nan=float(np.nanmedian(rdsm))), device=dev)
    barevalid = torch.tensor((~tree) & np.isfinite(rdsm), device=dev)
    ar = torch.tensor(arow, device=dev); ac = torch.tensor(acol, device=dev)
    av = torch.tensor(aval, device=dev, dtype=torch.float64)

    def tv(z):
        return (z[1:, :] - z[:-1, :]).abs().mean() + (z[:, 1:] - z[:, :-1]).abs().mean()

    def fit(use_lidar):
        Z = rdsm_t.clone().requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=0.2)
        for _ in range(args.iters):
            opt.zero_grad()
            loss = (args.w_ceil * torch.relu(Z - rdsm_t).mean()
                    + args.w_bare * ((Z - rdsm_t)[barevalid] ** 2).mean()
                    + args.w_tv * tv(Z))
            if use_lidar:
                loss = loss + args.w_lidar * torch.nn.functional.huber_loss(Z[ar, ac], av, delta=2.0)
            loss.backward(); opt.step()
        return Z.detach().cpu().numpy()

    Z_lidar = fit(True); Z_nolid = fit(False)
    tbl = {
        "EOGS surface as ground (baseline)": (mae(rdsm, dtm3dep, treevalid), mae(rdsm, dtm3dep, tall)),
        "Ground fit WITHOUT lidar (priors)": (mae(Z_nolid, dtm3dep, treevalid), mae(Z_nolid, dtm3dep, tall)),
        "Ground fit WITH GEDI lidar":        (mae(Z_lidar, dtm3dep, treevalid), mae(Z_lidar, dtm3dep, tall)),
    }
    log("========= RESULT: MAE vs 3DEP bare-earth DTM (m) =========")
    log(f"  {'method':38s} {'tree':>7s} {'tall-canopy':>12s}")
    for k, (mt, mtt) in tbl.items():
        log(f"  {k:38s} {mt:7.2f} {mtt:12.2f}")
    log("=========================================================")
    log(f"  (n_anchors in tile = {n_anch}; lower = closer to true ground)")

    np.savez(os.path.join(out, f"{scene}_ground_dtm.npz"),
             rdsm=rdsm, dtm3dep=dtm3dep, gt_dsm=gt_dsm, tree=tree, tall=tall,
             Z_lidar=Z_lidar, Z_nolid=Z_nolid, arow=arow, acol=acol, aval=aval,
             results=json.dumps({k: list(v) for k, v in tbl.items()}), n_anch=n_anch)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(15, 9))
        def show(a_, img, t, **k): im = a_.imshow(img, **k); a_.set_title(t); plt.colorbar(im, ax=a_, fraction=.046)
        show(ax[0, 0], rdsm, "EOGS surface (NAVD88)")
        show(ax[0, 1], dtm3dep, "3DEP bare-earth DTM")
        show(ax[0, 2], Z_lidar, "Ground fit WITH lidar")
        em = np.where(treevalid, np.abs(rdsm - dtm3dep), np.nan)
        el = np.where(treevalid, np.abs(Z_lidar - dtm3dep), np.nan)
        vmax = float(np.nanpercentile(em, 95)) if np.isfinite(em).any() else 1.0
        show(ax[1, 0], em, "|surface - DTM| (tree)", vmin=0, vmax=vmax)
        show(ax[1, 1], el, "|fit - DTM| (tree)", vmin=0, vmax=vmax)
        ax[1, 2].imshow(tree, cmap="Greys"); ax[1, 2].scatter(acol, arow, s=14, c="r")
        ax[1, 2].set_title(f"tree mask + {n_anch} anchors")
        plt.tight_layout(); fig.savefig(os.path.join(out, f"{scene}_ground_dtm.png"), dpi=110)
        log(f"figure -> {os.path.join(out, f'{scene}_ground_dtm.png')}")
    except Exception as e:
        log("figure skipped:", e)
    return 0

if __name__ == "__main__":
    sys.exit(main())
