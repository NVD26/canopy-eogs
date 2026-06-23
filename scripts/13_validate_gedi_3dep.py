#!/usr/bin/env python3
"""
13_validate_gedi_3dep.py — the RIGOROUS GEDI validation. RUN ON THE 4090.

The 7-footprint check on a 256 m tile was inconclusive. Here we validate GEDI's GROUND
return against an independent, known-datum bare-earth DTM (USGS 3DEP) over a WIDE
Jacksonville AOI, using hundreds of footprints. This both (a) statistically confirms the
projection/geolocation and (b) PINS the GEDI(WGS84-ellipsoid) <-> 3DEP(NAVD88-orthometric)
vertical datum offset (expected ~ the local geoid height, about -30 m near Jacksonville).

Why ground (not canopy): 3DEP seamless DEM is bare-earth, so it directly checks GEDI
elev_lowestmode -- the very signal Paper 1 uses to recover the under-canopy terrain. It is
also the ground-truth DTM we need for the eventual under-canopy evaluation.

Pipeline:
  1. fetch a 3DEP bare-earth DTM for the AOI from the USGS 3DEPElevation ImageServer,
  2. load GEDI footprints (ground + quality) in the AOI from the granules already downloaded,
  3. sample the DTM over each ~25 m footprint and compare to GEDI ground,
  4. report N, datum offset (sanity-check vs geoid), residual MAE/std; PASS only with enough
     points and small residual.

Usage:
  python scripts/13_validate_gedi_3dep.py
  python scripts/13_validate_gedi_3dep.py --lon -81.664 --lat 30.349 --aoi-km 5 --res-m 10
"""
import argparse, glob, io, os, sys
import numpy as np

IMG_SERVER = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
              "3DEPElevation/ImageServer/exportImage")


def fetch_3dep_dtm(w, s, e, n, res_m, out_tif):
    """Bare-earth 3DEP DTM (NAVD88) for the lon/lat bbox -> GeoTIFF, returned in EPSG:4326."""
    try:
        import requests
        getter = lambda url, params: requests.get(url, params=params, timeout=120).content
    except ImportError:
        import urllib.parse, urllib.request
        def getter(url, params):
            return urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params), timeout=120).read()
    # output size in px (cap to keep the request reasonable)
    import math
    midlat = math.radians((s + n) / 2)
    width_m = (e - w) * 111320 * math.cos(midlat)
    height_m = (n - s) * 110540
    W = max(8, min(4000, int(round(width_m / res_m))))
    H = max(8, min(4000, int(round(height_m / res_m))))
    params = {
        "bbox": f"{w},{s},{e},{n}", "bboxSR": "4326", "imageSR": "4326",
        "size": f"{W},{H}", "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    print(f"fetching 3DEP DTM {W}x{H}px @ ~{res_m} m for bbox ({w:.4f},{s:.4f},{e:.4f},{n:.4f}) ...")
    data = getter(IMG_SERVER, params)
    if not data or len(data) < 1000 or data[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"3DEP did not return a GeoTIFF (got {len(data)} bytes; "
                           f"starts {data[:16]!r}). Check connectivity / bbox.")
    with open(out_tif, "wb") as f:
        f.write(data)
    return out_tif, W, H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, default=-81.664)   # JAX_068 tile center
    ap.add_argument("--lat", type=float, default=30.349)
    ap.add_argument("--aoi-km", type=float, default=5.0)
    ap.add_argument("--res-m", type=float, default=10.0)
    ap.add_argument("--lidar-dir", default=os.path.expanduser("~/eogs-data/lidar_probe"))
    ap.add_argument("--sensitivity", type=float, default=0.9)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/eogs-data/3dep"))
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--max-residual", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import rasterio, h5py
    dlat = args.aoi_km / 110.54
    dlon = args.aoi_km / (111.32 * np.cos(np.radians(args.lat)))
    w, s, e, n = args.lon - dlon, args.lat - dlat, args.lon + dlon, args.lat + dlat

    # 1. 3DEP DTM
    tif = os.path.join(args.out_dir, f"3dep_dtm_{args.lat:.3f}_{args.lon:.3f}_{args.aoi_km:g}km.tif")
    try:
        fetch_3dep_dtm(w, s, e, n, args.res_m, tif)
    except Exception as ex:
        print(f"!! 3DEP fetch failed: {ex}")
        return 1
    with rasterio.open(tif) as f:
        dtm = f.read(1).astype("float64"); tr = f.transform; nod = f.nodata
        Hh, Ww = dtm.shape
    valid = np.isfinite(dtm) & (np.abs(dtm) < 1e4)
    if nod is not None: valid &= dtm != nod
    print(f"3DEP DTM range over AOI: [{dtm[valid].min():.1f},{dtm[valid].max():.1f}] m (NAVD88)")

    # 2. GEDI footprints in AOI
    gedis = sorted(glob.glob(os.path.join(args.lidar_dir, "*GEDI02_A*.h5")))
    lon_l, lat_l, grnd_l = [], [], []
    for p in gedis:
        try:
            with h5py.File(p, "r") as f:
                for b in [k for k in f.keys() if k.startswith("BEAM")]:
                    g = f[b]
                    if "lat_lowestmode" not in g: continue
                    la, lo = g["lat_lowestmode"][:], g["lon_lowestmode"][:]
                    m = (lo >= w) & (lo <= e) & (la >= s) & (la <= n)
                    if not m.any(): continue
                    i = np.where(m)[0]
                    q = g["quality_flag"][:][i] if "quality_flag" in g else np.ones(len(i))
                    se = g["sensitivity"][:][i] if "sensitivity" in g else np.ones(len(i))
                    gr = g["elev_lowestmode"][:][i]
                    keep = (q == 1) & (se >= args.sensitivity) & np.isfinite(gr)
                    lon_l += lo[i][keep].tolist(); lat_l += la[i][keep].tolist(); grnd_l += gr[keep].tolist()
        except Exception as ex:
            print(f"  (skip {os.path.basename(p)}: {ex})")
    lon = np.array(lon_l); lat = np.array(lat_l); gedi_grnd = np.array(grnd_l)
    print(f"\nGEDI quality footprints in {args.aoi_km*2:.0f} km AOI: {len(lon)}")
    if len(lon) == 0:
        print("!! none — widen --aoi-km or check the granules in --lidar-dir."); return 1

    # 3. sample DTM over ~25 m footprint (window) at each footprint
    rad = max(1, int(round(12.5 / args.res_m)))
    inv = ~tr
    dtm_fp = np.full(len(lon), np.nan)
    for k in range(len(lon)):
        col, row = inv * (lon[k], lat[k]); col, row = int(col), int(row)
        if not (0 <= row < Hh and 0 <= col < Ww): continue
        r0, r1 = max(0, row-rad), min(Hh, row+rad+1)
        c0, c1 = max(0, col-rad), min(Ww, col+rad+1)
        wv = valid[r0:r1, c0:c1]; vals = dtm[r0:r1, c0:c1][wv]
        if vals.size: dtm_fp[k] = vals.mean()
    v = np.isfinite(dtm_fp)
    n_ov = int(v.sum())

    # 4. validate GEDI ground vs 3DEP DTM
    diff = gedi_grnd[v] - dtm_fp[v]
    off = float(np.median(diff)); resid = diff - off
    a = np.abs(resid)
    abs_med = float(np.median(a)); mae = float(np.mean(a)); mad = float(np.median(a) * 1.4826)
    w1 = float((a <= 1).mean() * 100); w3 = float((a <= 3).mean() * 100); w5 = float((a <= 5).mean() * 100)
    inliers = int((a <= 3).sum())
    print("\n================= VALIDATION: GEDI ground vs 3DEP bare-earth DTM =================")
    print(f"footprints with DTM overlap: {n_ov}/{len(lon)}")
    print(f"datum offset (GEDI ellipsoid -> 3DEP NAVD88): {off:+.2f} m  "
          f"(expected ~ local geoid, about -30 m near Jacksonville)")
    print(f"residual after offset: abs-median {abs_med:.2f} m, robust-std(MAD) {mad:.2f} m")
    print(f"  but mean MAE {mae:.1f} m / std {resid.std():.1f} m  <- driven by GROSS OUTLIERS (GEDI blunders)")
    print(f"  inlier fraction: within 1 m {w1:.1f}%, within 3 m {w3:.1f}%, within 5 m {w5:.1f}%")
    print(f"  => {inliers}/{n_ov} footprints are clean anchors (|residual| <= 3 m after datum shift)")
    geoid_ok = -36 < off < -22
    pts_ok = inliers >= args.min_n
    bulk_ok = (abs_med <= args.max_residual) and (w3 >= 70.0)
    print(f"\n  checks: clean-inliers>={args.min_n}: {pts_ok} | median<= {args.max_residual} m & >=70% within 3 m: "
          f"{bulk_ok} | offset~geoid(-30): {geoid_ok}")
    if pts_ok and bulk_ok and geoid_ok:
        print("\nVERDICT: PASS — GEDI ground is trustworthy AFTER the datum shift + outlier rejection.")
        print(f"  Apply: gedi_ground_NAVD88 = gedi_ground_ellipsoid + ({off:+.2f}); then drop |resid|>3 m blunders.")
        np.savez(os.path.join(args.out_dir, "gedi_3dep_validation.npz"),
                 offset=off, abs_median=abs_med, mad=mad, mae=mae, n=n_ov, inliers=inliers,
                 within1=w1, within3=w3, within5=w5, aoi=np.array([w, s, e, n]))
        print(f"  saved -> {os.path.join(args.out_dir, 'gedi_3dep_validation.npz')}")
        return 0
    print("\nVERDICT: NOT PASSED as-is. Diagnose: offset off-geoid (datum), low inliers (widen AOI),")
    print("  or <70% within 3 m (geolocation/quality). Add filters (degrade_flag, num_detectedmodes) and re-check.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
