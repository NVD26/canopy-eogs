#!/usr/bin/env python3
"""
11_inspect_data.py — READ-ONLY inspection of the REAL data formats before we write
the GEDI anchor loader. Grounds the next step in what is actually on disk (no guessing
field names / datums / georeferencing). RUN ON THE 4090.

Dumps, for one scene (default JAX_068):
  1. Truth DSM georeferencing: the <scene>_DSM.txt 4-value metadata (xoff,yoff,size,res),
     the DSM raster shape/range, the EOGS tree-mask coverage, and the UTM zone.
  2. GEDI L2A contents: the dataset fields available under a BEAM group, and sample
     footprints inside the scene tile (ground vs canopy-top elevations + quality).

Nothing is written or trained. Paste the output back so the anchor loader + its
validation (GEDI canopy-top vs airborne DSM agreement) is built against real formats.

Usage:
  python scripts/11_inspect_data.py
  python scripts/11_inspect_data.py --scene JAX_068 --lidar-dir ~/eogs-data/lidar_probe
"""
import argparse, glob, os, sys
import numpy as np


def tile_center_lonlat(eogs_dir, scene):
    """Reuse RPC localization (same approach as 10_query_lidar) to get tile center."""
    import json
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
            lon = find({"lon_offset", "LONG_OFF", "long_off"})
            lat = find({"lat_offset", "LAT_OFF", "lat_off"})
            return lon, lat
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="JAX_068")
    ap.add_argument("--eogs-dir", default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--lidar-dir", default=os.path.expanduser("~/eogs-data/lidar_probe"))
    ap.add_argument("--pad-deg", type=float, default=0.004)
    args = ap.parse_args()
    sc = args.scene
    gt_dir = os.path.join(args.eogs_dir, "data", "truth", sc)

    print("=" * 72); print(f"PART 1 — truth DSM georeferencing for {sc}"); print("=" * 72)
    txt = os.path.join(gt_dir, f"{sc}_DSM.txt")
    if os.path.exists(txt):
        meta = np.loadtxt(txt)
        print(f"{sc}_DSM.txt -> {np.array(meta).ravel().tolist()}")
        print("  interpreted as (xoff_E, yoff_N, size_px, resolution_m) [UTM]")
    else:
        print(f"!! {txt} not found (IARPA scenes are geolocalized in the .tif instead)")
    try:
        import rasterio
        with rasterio.open(os.path.join(gt_dir, f"{sc}_DSM.tif")) as f:
            a = f.read(1).astype(float)
            print(f"DSM.tif shape={a.shape} valid range=[{np.nanmin(a):.1f},{np.nanmax(a):.1f}] crs={f.crs}")
    except Exception as e:
        print(f"  (DSM.tif read note: {e})")
    tm = os.path.join(args.eogs_dir, "scripts", "eval", "tree_masks", f"{sc}.png")
    if os.path.exists(tm):
        try:
            import imageio.v2 as imageio
            m = imageio.imread(tm)
        except Exception:
            import matplotlib.pyplot as plt; m = plt.imread(tm)
        m = np.asarray(m); mm = m[..., 0] if m.ndim == 3 else m
        frac = float((mm > 0).mean())
        print(f"tree_mask {sc}.png shape={mm.shape} tree-pixel fraction={frac*100:.1f}%")
    else:
        print(f"  (no tree mask at {tm})")
    lon, lat = tile_center_lonlat(args.eogs_dir, sc)
    if lon is not None:
        try:
            import utm
            e, n, zn, zl = utm.from_latlon(lat, lon)
            print(f"tile center lon/lat=({lon:.4f},{lat:.4f}) -> UTM zone {zn}{zl}  E={e:.1f} N={n:.1f}")
        except Exception as e:
            print(f"  (utm note: {e}; center lon/lat=({lon:.4f},{lat:.4f}))")
    bbox = (lon - args.pad_deg, lat - args.pad_deg, lon + args.pad_deg, lat + args.pad_deg) if lon else None

    print(); print("=" * 72); print("PART 2 — GEDI L2A fields + sample footprints in tile"); print("=" * 72)
    gedis = sorted(glob.glob(os.path.join(args.lidar_dir, "*GEDI02_A*.h5")) +
                   glob.glob(os.path.join(args.lidar_dir, "*GEDI02_A*.hdf5")))
    if not gedis:
        gedis = sorted(glob.glob(os.path.join(args.lidar_dir, "*.h5")))
    if not gedis:
        print(f"!! no GEDI .h5 found in {args.lidar_dir} (run 10_query_lidar.py --count-footprints first)")
        return 0
    import h5py
    p = gedis[0]
    print(f"granule: {os.path.basename(p)}")
    with h5py.File(p, "r") as f:
        beams = [k for k in f.keys() if k.startswith("BEAM")]
        print(f"beams: {beams}")
        if beams:
            g = f[beams[0]]
            keys = [k for k in g.keys()]
            print(f"\ntop-level datasets under {beams[0]} (first 40):")
            print("  " + ", ".join(keys[:40]))
            for sub in ("geolocation", "rx_processing_a1"):
                if sub in g and isinstance(g[sub], h5py.Group):
                    print(f"  [{sub}/]: " + ", ".join(list(g[sub].keys())[:20]))
        # sample footprints in bbox across all beams
        if bbox:
            rows = []
            for b in beams:
                g = f[b]
                if "lat_lowestmode" not in g:
                    continue
                lat_a, lon_a = g["lat_lowestmode"][:], g["lon_lowestmode"][:]
                m = ((lon_a >= bbox[0]) & (lon_a <= bbox[2]) &
                     (lat_a >= bbox[1]) & (lat_a <= bbox[3]))
                if not m.any():
                    continue
                idx = np.where(m)[0]
                def get(name):
                    return g[name][:][idx] if name in g else np.full(len(idx), np.nan)
                ground = get("elev_lowestmode"); top = get("elev_highestreturn")
                qual = get("quality_flag"); sens = get("sensitivity"); degr = get("degrade_flag")
                rh = g["rh"][:][idx] if "rh" in g else None
                rh100 = rh[:, 100] if rh is not None and rh.shape[1] > 100 else np.full(len(idx), np.nan)
                for j in range(len(idx)):
                    rows.append((b, lon_a[idx[j]], lat_a[idx[j]], ground[j], top[j],
                                 rh100[j], qual[j], sens[j], degr[j]))
            print(f"\nfootprints in tile (this 1 granule): {len(rows)}")
            print(f"{'beam':<8}{'lon':>10}{'lat':>9}{'ground':>9}{'canopytop':>11}{'rh100':>8}{'qual':>6}{'sens':>7}{'degr':>6}")
            for r in rows[:12]:
                print(f"{r[0]:<8}{r[1]:>10.4f}{r[2]:>9.4f}{r[3]:>9.2f}{r[4]:>11.2f}{r[5]:>8.2f}{r[6]:>6.0f}{r[7]:>7.2f}{r[8]:>6.0f}")
            if rows:
                g_arr = np.array([r[3] for r in rows]); t_arr = np.array([r[4] for r in rows])
                print(f"\nground elev range [{np.nanmin(g_arr):.1f},{np.nanmax(g_arr):.1f}] m; "
                      f"canopy height (top-ground) median {np.nanmedian(t_arr-g_arr):.1f} m")
    print("\nNote: GEDI elevations are WGS84-ellipsoid heights. Next step: the anchor loader will")
    print("project these into the scene UTM and CROSS-CHECK GEDI canopy-top vs the airborne DSM")
    print("(after a global datum offset) — only if they agree (~1-2 m) do we trust them as anchors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
