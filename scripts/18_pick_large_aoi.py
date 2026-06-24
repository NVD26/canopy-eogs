#!/usr/bin/env python3
"""
18_pick_large_aoi.py — [Paper 1 / M8] choose the best larger Jacksonville AOI for a scene
with ENOUGH GEDI ground anchors. Slides square windows of several sizes over the validated
anchor set and reports the window maximizing (a) total ground anchors and (b) tall-canopy
anchors (canopy height > 3 m). Output: best centre (UTM + lon/lat), counts, and a map.
"""
import os, sys, argparse, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--scene", default="JAX_068")
    ap.add_argument("--sizes", default="512,768,1024")
    ap.add_argument("--step", type=float, default=50.0)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anchors = args.anchors or os.path.join(repo, "data", "anchors", f"{args.scene}_gedi_anchors.npz")
    a = np.load(anchors, allow_pickle=True)
    E, N = a["E"].astype(float), a["N"].astype(float)
    canopy = (a["canopytop"].astype(float) - a["ground"].astype(float))
    tall = canopy > 3.0
    print(f"loaded {len(E)} anchors; {int(tall.sum())} with canopy>3 m; "
          f"extent E[{E.min():.0f},{E.max():.0f}] N[{N.min():.0f},{N.max():.0f}] "
          f"({E.max()-E.min():.0f} x {N.max()-N.min():.0f} m)")
    best = {}
    for S in [float(x) for x in args.sizes.split(",")]:
        half = S / 2.0
        cE = np.arange(E.min()+half, E.max()-half+1, args.step)
        cN = np.arange(N.min()+half, N.max()-half+1, args.step)
        bestc, bn, bt = None, -1, -1
        for ce in cE:
            for cn in cN:
                m = (np.abs(E-ce) <= half) & (np.abs(N-cn) <= half)
                n = int(m.sum()); t = int((m & tall).sum())
                if n > bn or (n == bn and t > bt):
                    bestc, bn, bt = (ce, cn), n, t
        best[S] = (bestc, bn, bt)
        # to lon/lat
        try:
            import pyproj
            tr = pyproj.Transformer.from_crs(str(a["utm_epsg"]), "EPSG:4326", always_xy=True)
            lon, lat = tr.transform(bestc[0], bestc[1])
        except Exception:
            lon = lat = float("nan")
        print(f"  AOI {int(S)} m: best centre UTM ({bestc[0]:.0f},{bestc[1]:.0f}) "
              f"lon/lat ({lon:.5f},{lat:.5f}) -> {bn} anchors ({bt} tall-canopy)")
    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(E, N, c=canopy, s=6, cmap="viridis", vmin=0, vmax=25)
        plt.colorbar(sc, label="canopy height (m)")
        for S, (c, n, t) in best.items():
            h = S/2
            ax.add_patch(plt.Rectangle((c[0]-h, c[1]-h), S, S, fill=False, ec="red", lw=1.5))
            ax.text(c[0]-h, c[1]+h+10, f"{int(S)}m: {n} anc", color="red", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlabel("UTM E (m)"); ax.set_ylabel("UTM N (m)")
        ax.set_title(f"{args.scene} GEDI ground anchors + candidate AOIs")
        out = os.path.join(repo, "results", f"large_aoi_{args.scene}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight"); print("figure ->", out)
    except Exception as e:
        print("figure skipped:", e)

if __name__ == "__main__":
    sys.exit(main())
