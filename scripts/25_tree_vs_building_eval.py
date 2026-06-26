#!/usr/bin/env python3
"""
25_tree_vs_building_eval.py — split EOGS DSM error into VEGETATION vs BUILDING (non-tree) pixels.
Reuses the eval's registered DSM (<scene>_rdsm.tif) on the GT grid + the shipped tree mask.
Reports tree fraction (so we know the scene is balanced) and MAE on each class.

  python scripts/25_tree_vs_building_eval.py --rdsm <out>/<scene>_rdsm.tif --scene JAX_214 --eogs-dir ~/eogs-src/EOGS
"""
import os, sys, argparse, numpy as np
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rdsm",required=True)
    ap.add_argument("--scene",required=True)
    ap.add_argument("--eogs-dir",default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--tag",default="")
    a=ap.parse_args()
    import rasterio
    gt=os.path.join(a.eogs_dir,"data","truth",a.scene,f"{a.scene}_DSM.tif")
    tm=os.path.join(a.eogs_dir,"scripts","eval","tree_masks",f"{a.scene}.png")
    for p in (a.rdsm,gt,tm):
        if not os.path.exists(p): print("!! missing:",p); return 2
    with rasterio.open(a.rdsm) as f: pred=f.read(1).astype(np.float64)
    with rasterio.open(gt) as f: g=f.read(1).astype(np.float64)
    with rasterio.open(tm) as f: tree=f.read(1)>0.5
    H=min(pred.shape[0],g.shape[0],tree.shape[0]); W=min(pred.shape[1],g.shape[1],tree.shape[1])
    pred,g,tree=pred[:H,:W],g[:H,:W],tree[:H,:W]
    pred[pred<-1e3]=np.nan; g[g<-1e3]=np.nan
    valid=np.isfinite(pred)&np.isfinite(g)
    treev=valid&tree; bldg=valid&(~tree)
    def mae(m): d=np.abs(pred-g)[m]; return float(np.nanmean(d)) if m.sum() else float("nan")
    print(f"  [{a.tag or os.path.basename(a.rdsm)}] tree-frac {tree.mean()*100:.0f}% | "
          f"VEG MAE {mae(treev):.3f} m ({int(treev.sum())} px) | BLDG MAE {mae(bldg):.3f} m ({int(bldg.sum())} px) | "
          f"overall {mae(valid):.3f} m")
if __name__=="__main__": sys.exit(main())
