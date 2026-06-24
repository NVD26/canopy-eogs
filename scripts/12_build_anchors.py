#!/usr/bin/env python3
"""
12_build_anchors.py — build TRUSTWORTHY GEDI lidar anchors for a scene. RUN ON THE 4090.

This is the upgraded builder. It uses the independently-validated USGS 3DEP bare-earth
DTM (see scripts/13) to do three things that earlier checks proved are necessary:
  1. OUTLIER REJECT (quality control): align GEDI to 3DEP only to test each footprint, and
     drop the ~10% gross blunders whose aligned ground is beyond --max-residual of 3DEP.
  2. KEEP THE EOGS FRAME: we SAVE the RAW GEDI ellipsoidal heights, because EOGS reconstructs
     in the satellite RPC frame which is WGS84-ellipsoidal too. (3DEP is NAVD88/orthometric;
     the ~-29.6 m geoid offset is recorded for converting to NAVD88 when evaluating vs 3DEP,
     but is NOT applied to the supervision heights, or it would inject a ~30 m bias in EOGS.)
  3. SAVE both returns per footprint (ground + canopy top) in UTM, ready for the EOGS loss.

Area of interest:
  - default: the exact scene tile bounds from <scene>_DSM.txt (the EOGS-evaluated area),
  - --aoi-km K: a K-by-K box around the scene centre instead (more footprints; use this
    when training EOGS over a larger reconstruction than a single 256 m DFC2019 tile).

Usage:
  python scripts/12_build_anchors.py --scene JAX_068
  python scripts/12_build_anchors.py --scene JAX_068 --aoi-km 2 --sensitivity 0.9
"""
import argparse, glob, json, os, sys
import numpy as np

EPSG_WGS84 = "EPSG:4326"
IMG_SERVER = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
              "3DEPElevation/ImageServer/exportImage")


def utm_epsg_from_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def tile_center_lonlat(eogs_dir, scene):
    for sub in ("rpcs", "images"):
        d = os.path.join(eogs_dir, "data", sub, scene)
        hits = [h for h in sorted(glob.glob(os.path.join(d, "*.json")))
                if os.path.basename(h) not in ("train.txt", "test.txt")]
        if hits:
            dd = json.load(open(hits[0]))
            def find(keys):
                st = [dd]
                while st:
                    c = st.pop()
                    if isinstance(c, dict):
                        for k, v in c.items():
                            if k in keys and isinstance(v, (int, float)): return float(v)
                            if isinstance(v, (dict, list)): st.append(v)
                    elif isinstance(c, list): st.extend(c)
                return None
            return (find({"lon_offset", "LONG_OFF", "long_off"}),
                    find({"lat_offset", "LAT_OFF", "lat_off"}))
    return None, None


def scene_bbox_lonlat(eogs_dir, scene, pyproj_to_wgs):
    """Exact tile bounds from <scene>_DSM.txt (UTM) converted to lon/lat."""
    txt = os.path.join(eogs_dir, "data", "truth", scene, f"{scene}_DSM.txt")
    if not os.path.exists(txt):
        return None
    xoff, yoff, size, res = (float(v) for v in np.loadtxt(txt).ravel()[:4])
    E0, E1, N0, N1 = xoff, xoff + size * res, yoff, yoff + size * res
    xs, ys = [E0, E1, E0, E1], [N0, N0, N1, N1]
    lons, lats = pyproj_to_wgs.transform(xs, ys)
    return (min(lons), min(lats), max(lons), max(lats))


def fetch_3dep_dtm(w, s, e, n, res_m, out_tif):
    try:
        import requests
        getter = lambda u, p: requests.get(u, params=p, timeout=120).content
    except ImportError:
        import urllib.parse, urllib.request
        getter = lambda u, p: urllib.request.urlopen(u + "?" + urllib.parse.urlencode(p), timeout=120).read()
    import math
    midlat = math.radians((s + n) / 2)
    W = max(8, min(4000, int(round((e - w) * 111320 * math.cos(midlat) / res_m))))
    H = max(8, min(4000, int(round((n - s) * 110540 / res_m))))
    params = {"bbox": f"{w},{s},{e},{n}", "bboxSR": "4326", "imageSR": "4326",
              "size": f"{W},{H}", "format": "tiff", "pixelType": "F32",
              "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    data = getter(IMG_SERVER, params)
    if not data or len(data) < 1000 or data[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"3DEP did not return a GeoTIFF (got {len(data)} bytes).")
    open(out_tif, "wb").write(data)
    return out_tif


def load_gedi(lidar_dir, w, s, e, n, sensitivity):
    import h5py
    cols = {k: [] for k in ("lon", "lat", "ground", "top", "rh100", "sens")}
    for p in sorted(glob.glob(os.path.join(lidar_dir, "*GEDI02_A*.h5"))):
        try:
            with h5py.File(p, "r") as f:
                for b in [k for k in f.keys() if k.startswith("BEAM")]:
                    g = f[b]
                    if "lat_lowestmode" not in g: continue
                    la, lo = g["lat_lowestmode"][:], g["lon_lowestmode"][:]
                    m = (lo >= w) & (lo <= e) & (la >= s) & (la <= n)
                    if not m.any(): continue
                    i = np.where(m)[0]
                    gv = lambda nm: g[nm][:][i] if nm in g else np.full(len(i), np.nan)
                    q = gv("quality_flag"); se = gv("sensitivity")
                    gr = gv("elev_lowestmode"); tp = gv("elev_highestreturn")
                    rh = g["rh"][:][i] if "rh" in g else None
                    rh100 = rh[:, 100] if rh is not None and rh.shape[1] > 100 else np.full(len(i), np.nan)
                    keep = (q == 1) & (se >= sensitivity) & np.isfinite(gr) & np.isfinite(tp)
                    cols["lon"] += lo[i][keep].tolist(); cols["lat"] += la[i][keep].tolist()
                    cols["ground"] += gr[keep].tolist(); cols["top"] += tp[keep].tolist()
                    cols["rh100"] += rh100[keep].tolist(); cols["sens"] += se[keep].tolist()
        except Exception as ex:
            print(f"  (skip {os.path.basename(p)}: {ex})")
    return {k: np.asarray(v, dtype="float64") for k, v in cols.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="JAX_068")
    ap.add_argument("--eogs-dir", default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--lidar-dir", default=os.path.expanduser("~/eogs-data/lidar_probe"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--aoi-km", type=float, default=None, help="override scene tile with a KxKm box")
    ap.add_argument("--res-m", type=float, default=10.0)
    ap.add_argument("--sensitivity", type=float, default=0.9)
    ap.add_argument("--max-residual", type=float, default=3.0, help="keep |aligned ground - 3DEP| <= this (m)")
    ap.add_argument("--min-clean", type=int, default=30)
    args = ap.parse_args()
    sc = args.scene
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "..", "data", "anchors")
    os.makedirs(out_dir, exist_ok=True)

    import rasterio
    from pyproj import Transformer

    lon_c, lat_c = tile_center_lonlat(args.eogs_dir, sc)
    if lon_c is None:
        print(f"!! could not locate scene {sc} centre (no RPC json)."); return 1
    epsg = utm_epsg_from_lonlat(lon_c, lat_c)
    to_wgs = Transformer.from_crs(epsg, EPSG_WGS84, always_xy=True)
    to_utm = Transformer.from_crs(EPSG_WGS84, epsg, always_xy=True)

    if args.aoi_km:
        dlat = args.aoi_km / 110.54; dlon = args.aoi_km / (111.32 * np.cos(np.radians(lat_c)))
        w, s, e, n = lon_c - dlon, lat_c - dlat, lon_c + dlon, lat_c + dlat
        aoi_desc = f"{args.aoi_km} km box around {sc} centre"
    else:
        bb = scene_bbox_lonlat(args.eogs_dir, sc, to_wgs)
        if bb is None:
            print(f"!! no {sc}_DSM.txt for exact bounds; pass --aoi-km instead."); return 1
        w, s, e, n = bb
        aoi_desc = f"exact {sc} eval tile"
    print(f"AOI: {aoi_desc}  lon/lat ({w:.4f},{s:.4f})-({e:.4f},{n:.4f})  UTM {epsg}")

    # 3DEP DTM for the AOI (the cleaning + datum reference)
    dtm_tif = os.path.join(out_dir, f"{sc}_3dep_dtm.tif")
    try:
        fetch_3dep_dtm(w, s, e, n, args.res_m, dtm_tif)
    except Exception as ex:
        print(f"!! 3DEP fetch failed: {ex}"); return 1
    with rasterio.open(dtm_tif) as f:
        dtm = f.read(1).astype("float64"); tr = f.transform
        nod = f.nodata; Hh, Ww = dtm.shape
    dvalid = np.isfinite(dtm) & (np.abs(dtm) < 1e4)
    if nod is not None: dvalid &= dtm != nod

    # GEDI in AOI
    g = load_gedi(args.lidar_dir, w, s, e, n, args.sensitivity)
    n_raw = len(g["lon"])
    if n_raw == 0:
        print("!! no quality GEDI footprints in AOI. Widen --aoi-km or relax --sensitivity."); return 1

    # sample 3DEP over each ~25 m footprint (mean of valid pixels in a 12.5 m radius)
    inv = ~tr; rad = max(1, int(round(12.5 / args.res_m)))
    dtm_fp = np.full(n_raw, np.nan)
    for k in range(n_raw):
        col, row = inv * (g["lon"][k], g["lat"][k]); col, row = int(col), int(row)
        if not (0 <= row < Hh and 0 <= col < Ww): continue
        r0, r1 = max(0, row-rad), min(Hh, row+rad+1); c0, c1 = max(0, col-rad), min(Ww, col+rad+1)
        vals = dtm[r0:r1, c0:c1][dvalid[r0:r1, c0:c1]]
        if vals.size: dtm_fp[k] = vals.mean()
    ov = np.isfinite(dtm_fp)

    # 1. align to 3DEP ONLY to test each footprint (geoid offset; not applied to saved heights)
    offset = float(np.median(g["ground"][ov] - dtm_fp[ov]))   # ~ -geoid (ellipsoid -> NAVD88)
    resid = (g["ground"] - offset) - dtm_fp

    # 2. outlier reject
    clean = ov & (np.abs(resid) <= args.max_residual)
    n_clean = int(clean.sum())
    print(f"GEDI footprints: {n_raw} quality -> {n_clean} clean "
          f"(|aligned ground - 3DEP| <= {args.max_residual} m)")
    print(f"geoid offset (ellipsoid -> NAVD88, for 3DEP eval only): {offset:+.2f} m")
    if n_clean:
        print(f"clean-set ground residual vs 3DEP: median {np.median(np.abs(resid[clean])):.2f} m")
        ch = g["top"][clean] - g["ground"][clean]
        print(f"canopy height (top - ground) over clean set: median {np.median(ch):.1f} m, "
              f"range [{ch.min():.1f},{ch.max():.1f}]")

    # 3. save (UTM coords + both returns in the 3DEP/NAVD88 frame)
    E, N = to_utm.transform(g["lon"][clean], g["lat"][clean])
    validated = n_clean >= args.min_clean
    out = os.path.join(out_dir, f"{sc}_gedi_anchors.npz")
    np.savez(out,
             lon=g["lon"][clean], lat=g["lat"][clean], E=np.asarray(E), N=np.asarray(N),
             ground=g["ground"][clean], canopytop=g["top"][clean],   # RAW ellipsoidal (EOGS frame)
             rh100=g["rh100"][clean], sensitivity=g["sens"][clean],
             height_frame="WGS84 ellipsoid (matches EOGS RPC)",
             geoid_offset_to_navd88=offset,                          # subtract for 3DEP/NAVD88 eval
             utm_epsg=epsg, aoi_lonlat=np.array([w, s, e, n]),
             max_residual=args.max_residual, validated=validated)
    tag = "VALIDATED" if validated else f"LOW-COUNT ({n_clean} < {args.min_clean}; consider --aoi-km)"
    print(f"\nSaved {n_clean} clean two-return anchors [{tag}] -> {out}")
    print("Each anchor: UTM (E,N) + ground & canopy-top in WGS84-ellipsoid (EOGS frame).")
    print(f"(For 3DEP/NAVD88 comparison, subtract geoid_offset {offset:+.2f} m.)")
    print("Next (M7): feed these to a lidar loss in EOGS (a global vertical offset to EOGS's")
    print("internal frame is absorbed during training; plan-position and relative heights are fixed).")
    return 0 if validated else 2


if __name__ == "__main__":
    sys.exit(main())
