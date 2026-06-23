#!/usr/bin/env python3
"""
10_query_lidar.py — measure the GEDI / ICESat-2 supervision available over each
EOGS scene tile. RUN ON THE 4090 (Earthdata login already in ~/.netrc).

For each scene it:
  1. derives the tile's lon/lat bbox from the truth DSM GeoTIFF (reprojected to EPSG:4326),
  2. searches GEDI L2A (GEDI02_A), GEDI L2B (GEDI02_B) and ICESat-2 ATL08 over that
     bbox for a date range, and reports granule counts,
  3. with --count-footprints K, downloads up to K granules per product and counts the
     actual footprints / land-segments that fall inside the bbox (the real density).

This quantifies how much lidar supervision Paper 1 actually has per tile BEFORE any
modeling — the key de-risking step from the design doc (§5).

Usage:
  python scripts/10_query_lidar.py
  python scripts/10_query_lidar.py --scenes JAX_004 JAX_214
  python scripts/10_query_lidar.py --start 2019-01-01 --end 2024-12-31
  python scripts/10_query_lidar.py --count-footprints 2     # heavier: downloads a few granules

Requires (already in the eogs env): earthaccess, rasterio, pyproj, h5py, numpy.
"""
import argparse
import os
import sys

DEFAULT_SCENES = ["JAX_004", "JAX_068", "JAX_214", "JAX_260", "IARPA_001", "IARPA_002", "IARPA_003"]


def find_dsm(eogs_dir, scene):
    """Locate a truth DSM for the scene (may be None; RPC fallback handles it)."""
    cands = [
        os.path.join(eogs_dir, "data", "truth", scene, f"{scene}_DSM.tif"),
        os.path.join(eogs_dir, "data", "truth", scene, f"{scene}_DSM.tiff"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    d = os.path.join(eogs_dir, "data", "truth", scene)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".tif", ".tiff")):
                return os.path.join(d, f)
    return None


def _bbox_from_dsm(dsm_path, pad_deg):
    """Use the DSM georeferencing if it has a valid CRS (some bundles strip it)."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(dsm_path) as ds:
        if ds.crs is None or ds.transform is None or ds.transform.is_identity:
            return None
        b = ds.bounds
        w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
    return (w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)


def _load_json(p):
    import json
    with open(p) as f:
        return json.load(f)


def _rpc_json_for_scene(eogs_dir, scene):
    """Find one RPC json for the scene (EOGS stores them under data/rpcs/<scene>/)."""
    import glob
    for sub in ("rpcs", "images"):
        d = os.path.join(eogs_dir, "data", sub, scene)
        hits = sorted(glob.glob(os.path.join(d, "*.json")))
        hits = [h for h in hits if os.path.basename(h) not in ("train.txt", "test.txt")]
        if hits:
            return hits[0]
    return None


def _center_from_rpc_dict(d):
    """Pull (lon_center, lat_center) from an RPC dict across common key namings."""
    # search nested dicts for the offset keys
    def find(keys):
        stack = [d]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if k in keys and isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                stack.extend(cur)
        return None
    lon = find({"lon_offset", "LONG_OFF", "longitudeOffset", "long_off"})
    lat = find({"lat_offset", "LAT_OFF", "latitudeOffset", "lat_off"})
    return lon, lat


def _bbox_from_rpc(rpc_path, pad_deg):
    """Tight bbox by localizing the RPC's image-extent corners (rpcm); else center +/- pad."""
    d = _load_json(rpc_path)
    # Best: rpcm corner localization (true tile polygon).
    try:
        import rpcm
        try:
            rpc = rpcm.RPCModel(d)
        except Exception:
            rpc = rpcm.RPCModel(d, dict_format="rpcm")
        alt = float(rpc.alt_offset)
        c0, c1 = rpc.col_offset - rpc.col_scale, rpc.col_offset + rpc.col_scale
        r0, r1 = rpc.row_offset - rpc.row_scale, rpc.row_offset + rpc.row_scale
        lons, lats = [], []
        for c in (c0, c1):
            for r in (r0, r1):
                lon, lat = rpc.localization(c, r, alt)
                lons.append(float(lon)); lats.append(float(lat))
        return (min(lons) - pad_deg, min(lats) - pad_deg, max(lons) + pad_deg, max(lats) + pad_deg)
    except Exception:
        pass
    # Fallback: center +/- pad from the RPC lon/lat offsets.
    lon, lat = _center_from_rpc_dict(d)
    if lon is None or lat is None:
        raise RuntimeError(f"could not read lon/lat from RPC {os.path.basename(rpc_path)}")
    return (lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg)


def tile_bbox_lonlat(eogs_dir, scene, dsm_path, pad_deg=0.003):
    """Return (west, south, east, north) in EPSG:4326. Tries DSM georef, then RPC."""
    if dsm_path:
        try:
            bb = _bbox_from_dsm(dsm_path, pad_deg)
            if bb:
                return bb, "dsm-crs"
        except Exception:
            pass
    rpc = _rpc_json_for_scene(eogs_dir, scene)
    if rpc:
        return _bbox_from_rpc(rpc, pad_deg), "rpc"
    raise RuntimeError(f"no georef source for {scene} (no DSM CRS, no RPC json found)")


def in_bbox(lat, lon, bbox):
    w, s, e, n = bbox
    return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)


def count_gedi_footprints(granule_paths, bbox):
    """Count GEDI footprints inside bbox (handles L2A and L2B geolocation layouts)."""
    import h5py, numpy as np
    total = 0
    for p in granule_paths:
        try:
            with h5py.File(p, "r") as f:
                for beam in [k for k in f.keys() if k.startswith("BEAM")]:
                    g = f[beam]
                    lat = lon = None
                    if "lat_lowestmode" in g and "lon_lowestmode" in g:          # L2A
                        lat, lon = g["lat_lowestmode"][:], g["lon_lowestmode"][:]
                    elif "geolocation" in g and "lat_lowestmode" in g["geolocation"]:  # L2B
                        lat = g["geolocation"]["lat_lowestmode"][:]
                        lon = g["geolocation"]["lon_lowestmode"][:]
                    if lat is None:
                        continue
                    total += int(np.count_nonzero(in_bbox(lat, lon, bbox)))
        except Exception as e:  # noqa: BLE001
            print(f"    (skip {os.path.basename(p)}: {e})")
    return total


def count_atl08_segments(granule_paths, bbox):
    """Count ICESat-2 ATL08 land-segments inside bbox across downloaded .h5 granules."""
    import h5py, numpy as np
    total = 0
    for p in granule_paths:
        try:
            with h5py.File(p, "r") as f:
                for beam in [k for k in f.keys() if k.startswith("gt")]:
                    base = f"{beam}/land_segments"
                    if base not in f:
                        continue
                    lat = f[f"{base}/latitude"][:]
                    lon = f[f"{base}/longitude"][:]
                    total += int(np.count_nonzero(in_bbox(lat, lon, bbox)))
        except Exception as e:  # noqa: BLE001
            print(f"    (skip {os.path.basename(p)}: {e})")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eogs-dir", default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--count-footprints", type=int, default=0,
                    help="download up to K granules per product and count in-bbox footprints (heavier)")
    ap.add_argument("--download-dir", default=os.path.expanduser("~/eogs-data/lidar_probe"))
    args = ap.parse_args()

    try:
        import earthaccess
    except ImportError:
        print("earthaccess not installed (activate the eogs env)."); return 1

    earthaccess.login(strategy="netrc")  # uses ~/.netrc from 02_earthdata_auth.py

    products = [("GEDI02_A", "GEDI L2A"), ("GEDI02_B", "GEDI L2B"), ("ATL08", "ICESat-2 ATL08")]
    print(f"{'scene':<10} {'GEDI02_A':>9} {'GEDI02_B':>9} {'ATL08':>9}   bbox(lon/lat)")
    print("-" * 78)

    grand = {p[0]: 0 for p in products}
    for scene in args.scenes:
        dsm = find_dsm(args.eogs_dir, scene)   # may be None -> RPC fallback handles it
        try:
            bbox, method = tile_bbox_lonlat(args.eogs_dir, scene, dsm)
        except Exception as e:
            print(f"{scene:<10} (bbox failed: {e})")
            continue
        counts = {}
        granules = {}
        for short, _ in products:
            try:
                res = earthaccess.search_data(short_name=short, bounding_box=bbox,
                                              temporal=(args.start, args.end))
            except Exception as e:  # noqa: BLE001
                res = []
                print(f"  (search failed for {short}: {e})")
            counts[short] = len(res)
            granules[short] = res
            grand[short] += len(res)
        print(f"{scene:<10} {counts['GEDI02_A']:>9} {counts['GEDI02_B']:>9} {counts['ATL08']:>9}   "
              f"[{method}] ({bbox[0]:.3f},{bbox[1]:.3f})-({bbox[2]:.3f},{bbox[3]:.3f})")

        if args.count_footprints > 0:
            os.makedirs(args.download_dir, exist_ok=True)
            for short, label in products:
                res = granules[short][: args.count_footprints]
                if not res:
                    continue
                paths = earthaccess.download(res, local_path=args.download_dir)
                if short.startswith("GEDI02"):
                    n = count_gedi_footprints(paths, bbox)
                elif short == "ATL08":
                    n = count_atl08_segments(paths, bbox)
                else:
                    n = 0
                ng = max(len(paths), 1)
                est = int(round(n / ng * counts[short]))   # extrapolate to all granules
                print(f"    {label}: {n} in-bbox across {len(paths)} granule(s)"
                      f"  ~est. {est} over all {counts[short]} granules (2019-24)")

    print("-" * 78)
    print(f"{'TOTAL':<10} {grand['GEDI02_A']:>9} {grand['GEDI02_B']:>9} {grand['ATL08']:>9}   (granules)")
    print("\nNote: granule counts = orbit segments intersecting the tile; use --count-footprints")
    print("to measure the actual in-tile footprint density (the number that matters for supervision).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
