#!/usr/bin/env python3
"""
23_score_ground.py — [Paper 1 / M9] score the rendered two-surface GROUND DSM vs USGS 3DEP.
Reads the top-down TOP DSM (dsm/) and GROUND DSM (ground_dsm/) that the patched render.py writes,
datum-aligns the EOGS frame to 3DEP on bare pixels, and reports MAE vs 3DEP on tall-canopy pixels
and on pixels far from any GEDI anchor (generalization).

  python render.py -m <exp>          # first, with the two-surface patch applied (writes ground_dsm)
  python scripts/23_score_ground.py --model-path <exp> --tile JAX_113
"""
import os, sys, glob, json, argparse, urllib.request, urllib.parse
import numpy as np

USGS=("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage")
def fetch_3dep(w,s,e,n,W,H,out):
    p={"bbox":f"{w},{s},{e},{n}","bboxSR":"4326","imageSR":"4326","size":f"{W},{H}","format":"tiff",
       "pixelType":"F32","interpolation":"RSP_BilinearInterpolation","f":"image"}
    d=urllib.request.urlopen(USGS+"?"+urllib.parse.urlencode(p),timeout=180).read()
    if d[:2] not in (b"II",b"MM"): raise RuntimeError(f"3DEP not GeoTIFF ({len(d)} b)")
    open(out,"wb").write(d); return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path",required=True)
    ap.add_argument("--tile",default="JAX_113")
    ap.add_argument("--eogs-dir",default=os.path.expanduser("~/eogs-src/EOGS"))
    ap.add_argument("--anchors",default=None)
    ap.add_argument("--iters",type=int,default=5000)
    ap.add_argument("--far-m",type=float,default=30.0)
    a=ap.parse_args()
    repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anchors=a.anchors or os.path.join(repo,"data","anchors",f"{a.tile}_gedi_anchors.npz")
    import rasterio
    from rasterio.warp import reproject,Resampling,transform_bounds
    from scipy.ndimage import distance_transform_edt

    base=os.path.join(a.model_path,f"test_opNone/ours_{a.iters}")
    dsms=sorted(glob.glob(os.path.join(base,"dsm","*")))
    if not dsms: print("!! no top DSM in",base); return 2
    name=os.path.basename(dsms[-1])                      # top-down view = last
    gpath=os.path.join(base,"ground_dsm",name)
    if not os.path.exists(gpath):
        print("!! no ground_dsm/"+name+" — re-run `python render.py -m <exp>` with the two-surface patch."); return 2
    with rasterio.open(dsms[-1]) as f: top=f.read(1).astype(np.float64); tr=f.transform; crs=f.crs; H,W=top.shape
    with rasterio.open(gpath) as f: grd=f.read(1).astype(np.float64)
    top[top<-1e3]=np.nan; grd[grd<-1e3]=np.nan
    an=np.load(anchors,allow_pickle=True); geoid=float(an["geoid_offset_to_navd88"])

    # 3DEP over the DSM bounds, reprojected to the DSM grid
    b=rasterio.transform.array_bounds(H,W,tr)            # (w,s,e,n) in UTM
    wl,sl,el,nl=transform_bounds(crs,"EPSG:4326",*b)
    tif=os.path.join(a.model_path,"_3dep.tif"); fetch_3dep(wl,sl,el,nl,W,H,tif)
    dtm=np.full((H,W),np.nan)
    with rasterio.open(tif) as src:
        reproject(rasterio.band(src,1),dtm,src_transform=src.transform,src_crs=src.crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    dtm[dtm<-1e3]=np.nan

    # datum-align EOGS (ellipsoidal) to 3DEP (NAVD88) on BARE pixels (smallest top-3DEP residual)
    top0=top-geoid; resid=top0-dtm
    v=np.isfinite(resid)
    thr=np.nanpercentile(resid[v],30)                   # lowest 30% residual ~ bare ground
    offset=float(np.nanmedian(resid[v & (resid<=thr)]))
    topN=top0-offset; grdN=grd-geoid-offset
    print(f"[datum] geoid {geoid:+.2f}, bare-align offset {offset:+.2f} m")

    canopy=topN-dtm; valid=np.isfinite(topN)&np.isfinite(grdN)&np.isfinite(dtm)
    tall=valid&(canopy>3.0)
    # distance to nearest anchor (generalization)
    aE=an["E"].astype(float); aN=an["N"].astype(float)
    inv=~np.array(tr.is_identity)
    cinv=np.array([[ -tr.a, -tr.b, -tr.c],[-tr.d,-tr.e,-tr.f]])  # not used; compute index directly
    col=((aE-tr.c)/tr.a).astype(int); rowi=((aN-tr.f)/tr.e).astype(int)
    m=(col>=0)&(col<W)&(rowi>=0)&(rowi<H)
    amask=np.zeros((H,W),bool); amask[rowi[m],col[m]]=True
    dist=distance_transform_edt(~amask)*abs(tr.a)
    far=tall&(dist>a.far_m)
    def mae(x,msk): d=np.abs((x-dtm)[msk]); return float(np.nanmean(d)) if d.size else float("nan")
    # --- INTERPOLATION CONTROLS: does the optical fusion beat just interpolating the sparse lidar? ---
    from scipy.interpolate import griddata
    g_navd = an["ground"].astype(float) - geoid            # GEDI ground in NAVD88 (independent of 3DEP)
    pts = np.column_stack([aE[m], aN[m]]); vals = g_navd[m]
    GXr, GYr = np.meshgrid(tr.c + (np.arange(W)+0.5)*tr.a, tr.f + (np.arange(H)+0.5)*tr.e)
    def interp(P, V):
        lin = griddata(P, V, (GXr, GYr), method="linear")
        nz  = griddata(P, V, (GXr, GYr), method="nearest")
        bad = ~np.isfinite(lin); lin[bad] = nz[bad]; return lin
    gedi_interp = interp(pts, vals)                         # control A: interpolate GEDI ground only
    ca = topN[rowi[m], col[m]] - vals; ok2 = np.isfinite(ca)
    ground_B = topN - interp(pts[ok2], ca[ok2])            # control B: optical top - interpolated canopy height
    print("\n========= M9: GROUND vs 3DEP bare-earth (NAVD88), rendered surfaces =========")
    print(f"  valid {int(valid.sum())} px; tall-canopy {int(tall.sum())}; far-from-anchor tall {int(far.sum())}")
    print(f"  median canopy on tall pixels: {np.nanmedian(canopy[tall]):.1f} m")
    print(f"  {'':40s}{'all tall':>10s}{'far (gen.)':>12s}")
    print(f"  {'TOP surface as ground (baseline)':40s}{mae(topN,tall):10.2f}{mae(topN,far):12.2f}")
    print(f"  {'GEDI ground interpolated (no optical)':40s}{mae(gedi_interp,tall):10.2f}{mae(gedi_interp,far):12.2f}")
    print(f"  {'optical TOP - interp canopy (strong ctrl)':40s}{mae(ground_B,tall):10.2f}{mae(ground_B,far):12.2f}")
    print(f"  {'LEARNED ground (two-surface, ours)':40s}{mae(grdN,tall):10.2f}{mae(grdN,far):12.2f}")
    print("  (lower=better. Ours must beat both interpolation controls, ESPECIALLY far-from-anchor, to justify the method.)")
    print("============================================================================")
    np.savez(os.path.join(a.model_path,"m9_eval.npz"),top=topN,ground=grdN,dtm=dtm,tall=tall,far=far,
             gedi_interp=gedi_interp,ground_B=ground_B)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,3,figsize=(15,9))
        def sh(x,im,t,**k): c=x.imshow(im,**k);x.set_title(t);plt.colorbar(c,ax=x,fraction=.046)
        vm=float(np.nanpercentile(np.abs(topN-dtm)[tall],95)) if tall.any() else 15
        sh(ax[0,0],topN,"TOP (NAVD88)");sh(ax[0,1],grdN,"LEARNED ground");sh(ax[0,2],dtm,"3DEP bare-earth")
        sh(ax[1,0],np.where(tall,np.abs(topN-dtm),np.nan),"|top-3DEP| tall",vmin=0,vmax=vm)
        sh(ax[1,1],np.where(tall,np.abs(grdN-dtm),np.nan),"|ground-3DEP| tall",vmin=0,vmax=vm)
        sh(ax[1,2],np.where(tall,canopy,np.nan),"canopy height",vmin=0,vmax=vm)
        plt.tight_layout();fig.savefig(os.path.join(a.model_path,"m9_eval.png"),dpi=110)
        print("figure ->",os.path.join(a.model_path,"m9_eval.png"))
    except Exception as e: print("figure skipped:",e)

if __name__=="__main__": sys.exit(main())
