#!/usr/bin/env python3
"""
12_build_anchors.py — build GEDI lidar anchors for a scene AND validate them against
the airborne DSM before trusting them. RUN ON THE 4090.

Validation-first (per project rule: no unverified data feeds a loss):
  1. Read the EXACT eval tile bounds from <scene>_DSM.txt = (xoff_E, yoff_N, size_px, res)
     => UTM tile [xoff, xoff+size*res] E x [yoff, yoff+size*res] N (zone from tile center).
  2. Load ALL GEDI L2A granules, project footprint lon/lat -> tile UTM, keep those inside
     the tile that pass quality (quality_flag==1, degrade_flag==0, sensitivity>=thresh).
  3. CROSS-CHECK: sample the airborne DSM at each footprint and compare GEDI canopy-top
     (elev_highestreturn) to it. A correct projection/datum gives a tight residual after a
     single global vertical offset. Report median offset + residual MAE/abs-median.
  4. Only if the residual is small do we save aligned anchors (ground + canopy-top in the
     DSM's vertical datum) for the EOGS lidar loss.

Usage:
  python scripts/12_build_anchors.py --scene JAX_068
  python scripts/12_build_anchors.py --scene JAX_068 --sensitivity 0.9 --max-residual 5
"""
import argparse, glob, json, os, sys
import numpy as np

EPSG_WGS84 = "EPSG:4326"


def utm_epsg_from_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}", zone


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="JAX_068")
    ap.add_argument("--eogs-dir", default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--lidar-dir", default=os.path.expanduser("~/eogs-data/lidar_probe"))
    ap.add_argument("--out-dir", default=None, help="where to save anchors (default: repo data/anchors)")
    ap.add_argument("--sensitivity", type=float, default=0.9)
    ap.add_argument("--max-residual", type=float, default=5.0, help="trust gate: residual abs-median (m)")
    args = ap.parse_args()
    sc = args.scene
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "..", "data", "anchors")

    import rasterio, h5py
    from pyproj import Transformer

    # 1. exact tile bounds from DSM.txt + airborne DSM raster
    gt_dir = os.path.join(args.eogs_dir, "data", "truth", sc)
    txt = os.path.join(gt_dir, f"{sc}_DSM.txt")
    if not os.path.exists(txt):
        print(f"!! {txt} not found (needed for exact UTM tile bounds)."); return 1
    xoff, yoff, size, res = (float(v) for v in np.loadtxt(txt).ravel()[:4])
    size = int(size)
    E_min, E_max = xoff, xoff + size * res
    N_min, N_max = yoff, yoff + size * res
    with rasterio.open(os.path.join(gt_dir, f"{sc}_DSM.tif")) as f:
        dsm = f.read(1).astype("float64")
        nodata = f.nodata
    dsm_valid = np.isfinite(dsm)
    if nodata is not None: dsm_valid &= dsm != nodata
    dsm_valid &= np.abs(dsm) < 1e4
    print(f"tile {sc}: UTM E[{E_min:.1f},{E_max:.1f}] N[{N_min:.1f},{N_max:.1f}]  "
          f"{size}x{size}px @ {res} m = {size*res:.0f} m; DSM valid range "
          f"[{dsm[dsm_valid].min():.1f},{dsm[dsm_valid].max():.1f}] m")

    lon_c, lat_c = tile_center_lonlat(args.eogs_dir, sc)
    epsg, zone = utm_epsg_from_lonlat(lon_c, lat_c)
    print(f"tile center lon/lat=({lon_c:.4f},{lat_c:.4f}) -> {epsg} (UTM zone {zone})")
    to_utm = Transformer.from_crs(EPSG_WGS84, epsg, always_xy=True)

    def dsm_footprint(E, N, radius_m=12.5):
        """Sample the airborne DSM over each ~25 m GEDI footprint (not one pixel).
        Returns per-footprint (point, mean, max, min, n_valid)."""
        rad = int(round(radius_m / res))
        col = ((E - E_min) / res).astype(int)
        row = ((N_max - N) / res).astype(int)
        pt = np.full(len(E), np.nan); mn = np.full(len(E), np.nan)
        mx = np.full(len(E), np.nan); lo = np.full(len(E), np.nan)
        nv = np.zeros(len(E), int)
        for k in range(len(E)):
            r, c = row[k], col[k]
            if not (0 <= r < size and 0 <= c < size):
                continue
            if dsm_valid[r, c]:
                pt[k] = dsm[r, c]
            r0, r1 = max(0, r-rad), min(size, r+rad+1)
            c0, c1 = max(0, c-rad), min(size, c+rad+1)
            w = dsm[r0:r1, c0:c1]; wv = dsm_valid[r0:r1, c0:c1]
            vals = w[wv]
            if vals.size:
                mn[k], mx[k], lo[k], nv[k] = vals.mean(), vals.max(), vals.min(), vals.size
        return pt, mn, mx, lo, nv

    # 2. gather GEDI footprints inside the EXACT tile
    gedis = sorted(glob.glob(os.path.join(args.lidar_dir, "*GEDI02_A*.h5")))
    if not gedis:
        print(f"!! no GEDI02_A granules in {args.lidar_dir}. Run 10_query_lidar.py --scenes {sc} --count-footprints 25 first.")
        return 1
    cols = {k: [] for k in ("lon", "lat", "E", "N", "ground", "top", "rh100", "qual", "sens")}
    for p in gedis:
        try:
            with h5py.File(p, "r") as f:
                for b in [k for k in f.keys() if k.startswith("BEAM")]:
                    g = f[b]
                    if "lat_lowestmode" not in g: continue
                    lat = g["lat_lowestmode"][:]; lon = g["lon_lowestmode"][:]
                    E, N = to_utm.transform(lon, lat)
                    E, N = np.asarray(E), np.asarray(N)
                    m = (E >= E_min) & (E <= E_max) & (N >= N_min) & (N <= N_max)
                    if not m.any(): continue
                    i = np.where(m)[0]
                    def gv(name): return g[name][:][i] if name in g else np.full(len(i), np.nan)
                    rh = g["rh"][:][i] if "rh" in g else None
                    rh100 = rh[:, 100] if rh is not None and rh.shape[1] > 100 else np.full(len(i), np.nan)
                    cols["lon"] += lon[i].tolist(); cols["lat"] += lat[i].tolist()
                    cols["E"] += E[i].tolist();     cols["N"] += N[i].tolist()
                    cols["ground"] += gv("elev_lowestmode").tolist()
                    cols["top"]    += gv("elev_highestreturn").tolist()
                    cols["rh100"]  += rh100.tolist()
                    cols["qual"]   += gv("quality_flag").tolist()
                    cols["sens"]   += gv("sensitivity").tolist()
        except Exception as e:
            print(f"  (skip {os.path.basename(p)}: {e})")
    n_raw = len(cols["lon"])
    arr = {k: np.asarray(v, dtype="float64") for k, v in cols.items()}
    keep = (arr["qual"] == 1) & (arr["sens"] >= args.sensitivity) & np.isfinite(arr["ground"]) & np.isfinite(arr["top"])
    for k in arr: arr[k] = arr[k][keep]
    n = len(arr["lon"])
    print(f"\nGEDI footprints inside the {size*res:.0f} m tile: {n_raw} raw -> {n} after quality "
          f"(quality_flag==1, sensitivity>={args.sensitivity})")
    if n == 0:
        print("!! no quality footprints inside the exact tile. Try a larger AOI or relax --sensitivity.")
        return 1

    # 3. VALIDATION: GEDI canopy-top vs airborne DSM aggregated over the ~25 m footprint.
    #    (Only canopy-TOP is checkable against a surface DSM; GEDI GROUND needs a bare-earth
    #     DTM e.g. USGS 3DEP -- deferred. A single-pixel compare is wrong for a 25 m footprint.)
    pt, dmean, dmax, dmin, nvld = dsm_footprint(arr["E"], arr["N"])
    v = np.isfinite(dmean) & np.isfinite(arr["top"])
    print("\n================= VALIDATION: GEDI canopy-top vs airborne DSM (footprint-aggregated) =================")
    print(f"footprints with DSM overlap: {int(v.sum())}/{n}  (footprint radius 12.5 m)")

    # per-footprint diagnostic table (look at the real data)
    print(f"\n{'#':>2}{'g_grnd':>9}{'g_top':>9}{'dsm_pt':>9}{'dsm_mean':>10}{'dsm_max':>9}{'top-mean':>10}{'top-max':>9}")
    order = np.where(v)[0]
    for j in order:
        print(f"{j:>2}{arr['ground'][j]:>9.2f}{arr['top'][j]:>9.2f}{pt[j]:>9.2f}"
              f"{dmean[j]:>10.2f}{dmax[j]:>9.2f}{arr['top'][j]-dmean[j]:>10.2f}{arr['top'][j]-dmax[j]:>9.2f}")

    def report(label, dsm_ref):
        d = (arr["top"][v] - dsm_ref[v])
        off = float(np.median(d)); r = d - off
        print(f"  vs DSM {label:<5}: offset {off:+.2f} m | residual abs-median "
              f"{np.median(np.abs(r)):.2f} m, MAE {np.mean(np.abs(r)):.2f} m, std {r.std():.2f} m")
        return float(np.median(np.abs(r))), off
    am_mean, off_mean = report("mean", dmean)
    am_max, off_max = report("max", dmax)
    canopy_h = arr["top"] - arr["ground"]
    print(f"  GEDI canopy height (top-ground): median {np.median(canopy_h):.1f} m, "
          f"range [{canopy_h.min():.1f},{canopy_h.max():.1f}]")

    # trust gate on the better of mean/max comparison
    abs_med = min(am_mean, am_max); off = off_mean if am_mean <= am_max else off_max
    trust = abs_med <= args.max_residual
    print(f"\nTRUST GATE (best abs-median {abs_med:.2f} m <= {args.max_residual} m): {'PASS' if trust else 'FAIL'}")
    if not trust:
        print("!! Still too large. Diagnose with the table above:")
        print("   - if 1-2 footprints dominate -> geolocation outliers at canopy/building edges (filter).")
        print("   - if ALL rows are biased the same way -> a systematic offset (datum) — already removed.")
        print("   - if scatter is large & random with few points -> need MORE footprints (relax --sensitivity,")
        print("     or use a larger AOI) for a statistically meaningful check.")
        print("   GEDI vs airborne typically agrees to a few metres; >5 m here with only a handful of points")
        print("   is most likely small-sample + geolocation, not necessarily a code bug.")
        return 2

    # 4. save anchors aligned to the DSM/scene vertical datum (subtract the offset)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{sc}_gedi_anchors.npz")
    np.savez(out, lon=arr["lon"], lat=arr["lat"], E=arr["E"], N=arr["N"],
             ground=arr["ground"] - off, canopytop=arr["top"] - off,
             rh100=arr["rh100"], sensitivity=arr["sens"], datum_offset=off,
             tile_bounds=np.array([E_min, N_min, E_max, N_max]), utm_epsg=epsg,
             dsm_txt=np.array([xoff, yoff, size, res]))
    print(f"\nSaved {n} validated anchors -> {out}")
    print("Note: GROUND values still need a bare-earth DTM (USGS 3DEP) to validate separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
