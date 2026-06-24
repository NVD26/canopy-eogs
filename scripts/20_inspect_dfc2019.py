#!/usr/bin/env python3
"""
20_inspect_dfc2019.py — [Paper 1 / M8] map the DFC2019 JAX tile layout + GEDI coverage to pick
how to build a denser scene. RGB are per-tile 256 m crops (JAX_<tile>_<view>_RGB.tif), so we
need a CONTIGUOUS cluster of tiles (or full strips). Reports per JAX tile: UTM bbox, #views,
#GEDI ground anchors (from our validated set), and which tiles are spatially adjacent.
"""
import os, sys, glob, argparse, numpy as np
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfc-dir", default=os.path.expanduser("~/eogs-src/DFC2019"))
    ap.add_argument("--rgb-dirs", default="Track3-RGB-1,Track3-RGB-2")
    ap.add_argument("--anchors", default=None)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anchors = args.anchors or os.path.join(repo, "data", "anchors", "JAX_068_gedi_anchors.npz")
    a = np.load(anchors, allow_pickle=True)
    E, N = a["E"].astype(float), a["N"].astype(float)

    truth = os.path.join(args.dfc_dir, "Track3-Truth")
    txts = sorted(glob.glob(os.path.join(truth, "JAX_*_DSM.txt")))
    if not txts:
        print(f"!! no JAX_*_DSM.txt in {truth}"); return 2

    # count views per tile across both RGB dirs (ignore :Zone.Identifier ADS files)
    views = defaultdict(set)
    for d in args.rgb_dirs.split(","):
        for p in glob.glob(os.path.join(args.dfc_dir, d, "JAX_*_RGB.tif")):
            b = os.path.basename(p)
            if ":" in b: continue
            parts = b.split("_")            # JAX_<tile>_<view>_RGB.tif
            if len(parts) >= 4:
                views[f"JAX_{parts[1]}"].add(parts[2])

    tiles = {}
    for t in txts:
        tile = os.path.basename(t).replace("_DSM.txt", "")
        roi = np.loadtxt(t); xoff, yoff, size, res = roi[0], roi[1], int(roi[2]), roi[3]
        bb = (xoff, xoff+size*res, yoff, yoff+size*res)
        n_anch = int(((E>=bb[0])&(E<=bb[1])&(N>=bb[2])&(N<=bb[3])).sum())
        tiles[tile] = dict(bb=bb, nv=len(views.get(tile, [])), na=n_anch)

    # tiles overlapping our validated anchor region
    inset = {k: v for k, v in tiles.items() if v["na"] > 0}
    print(f"{len(tiles)} JAX truth tiles total; {len(inset)} carry GEDI anchors (our validated set).")
    print("\nTiles WITH anchors (sorted):")
    for k, v in sorted(inset.items(), key=lambda kv: -kv[1]["na"]):
        print(f"  {k}: {v['na']:3d} anchors | {v['nv']:2d} views | UTM E[{v['bb'][0]:.0f},{v['bb'][1]:.0f}] N[{v['bb'][2]:.0f},{v['bb'][3]:.0f}]")

    # adjacency among anchor-bearing tiles (bboxes touch within 5 m)
    def adj(b1, b2, tol=5.0):
        return (b1[0] <= b2[1]+tol and b2[0] <= b1[1]+tol and
                b1[2] <= b2[3]+tol and b2[2] <= b1[3]+tol)
    keys = list(inset)
    print("\nAdjacent anchor-tile pairs (could mosaic into a bigger scene):")
    found = False
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            if adj(inset[keys[i]]["bb"], inset[keys[j]]["bb"]):
                print(f"  {keys[i]} <-> {keys[j]}  (combined {inset[keys[i]]['na']+inset[keys[j]]['na']} anchors)")
                found = True
    if not found:
        print("  none adjacent — anchor-bearing tiles are spatially separated.")
    # best single tile
    best = max(tiles.items(), key=lambda kv: kv[1]["na"])
    print(f"\nBest SINGLE tile by anchors: {best[0]} ({best[1]['na']} anchors, {best[1]['nv']} views).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
