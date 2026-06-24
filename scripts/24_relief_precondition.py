#!/usr/bin/env python3
"""
24_relief_precondition.py — [Paper 1 / worth-it check] BEFORE building anything on new imagery,
test the precondition for the whole idea: does GEDI-density interpolation FAIL on relief terrain?

Pure data experiment (3DEP only; no GEDI download, no imagery, no GPU). Over a given AOI it:
  1. fetches the 3DEP bare-earth DTM (truth),
  2. subsamples it at a GEDI-REALISTIC density (random, ~JAX accumulated density; optimistic vs
     real track-clustered GEDI), then interpolates back,
  3. reports interpolation MAE vs the true DTM and the terrain ruggedness (slope std).
Run on a FLAT and a RELIEF forested AOI. If interpolation MAE grows sharply with ruggedness, there
is headroom the optical surface could fill -> worth building. If interpolation stays small even on
relief, GEDI alone suffices everywhere -> the fusion idea is not worth pursuing.

  python scripts/24_relief_precondition.py --name SmokyMtns --bbox -83.55 35.55 -83.45 35.65
  python scripts/24_relief_precondition.py --name Jacksonville --bbox -81.70 30.33 -81.66 30.37
"""
import os, sys, argparse, urllib.request, urllib.parse
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
    ap.add_argument("--name",required=True)
    ap.add_argument("--bbox",nargs=4,type=float,required=True,metavar=("W","S","E","N"))
    ap.add_argument("--res-m",type=float,default=10.0,help="grid resolution for the test (m)")
    ap.add_argument("--gedi-per-km2",type=float,default=330.0,help="accumulated GEDI footprint density (JAX-like)")
    ap.add_argument("--reps",type=int,default=5)
    args=ap.parse_args()
    repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out=os.path.join(repo,"results","relief_precondition"); os.makedirs(out,exist_ok=True)
    import rasterio
    from rasterio.warp import transform_bounds
    from scipy.interpolate import griddata
    import pyproj

    w,s,e,n=args.bbox
    # size in pixels at res-m (approx via local UTM)
    midlat=(s+n)/2; mlon=(w+e)/2
    utm_zone=int((mlon+180)//6)+1; epsg=f"EPSG:{32600+utm_zone if midlat>=0 else 32700+utm_zone}"
    tr=pyproj.Transformer.from_crs("EPSG:4326",epsg,always_xy=True)
    x0,y0=tr.transform(w,s); x1,y1=tr.transform(e,n)
    Wm=abs(x1-x0); Hm=abs(y1-y0); W=int(Wm/args.res_m); H=int(Hm/args.res_m)
    W=min(W,2000); H=min(H,2000)
    tif=os.path.join(out,f"{args.name}_dtm.tif"); fetch_3dep(w,s,e,n,W,H,tif)
    with rasterio.open(tif) as f: dtm=f.read(1).astype(np.float64)
    dtm[dtm<-1e3]=np.nan
    area_km2=(Wm*Hm)/1e6
    # ruggedness: std of elevation + std of local slope
    gy,gx=np.gradient(dtm, args.res_m)
    slope=np.sqrt(gx**2+gy**2)
    relief=float(np.nanmax(dtm)-np.nanmin(dtm)); slope_std=float(np.nanstd(slope))*100
    print(f"[{args.name}] {W}x{H}@{args.res_m}m  ~{area_km2:.1f} km^2  relief {relief:.0f} m  "
          f"elev-std {np.nanstd(dtm):.1f} m  slope-std {slope_std:.1f}%")

    valid=np.isfinite(dtm); idx=np.argwhere(valid)
    npts=max(20,int(args.gedi_per_km2*area_km2))
    YY,XX=np.mgrid[0:H,0:W]
    maes=[]
    rng=np.random.default_rng(0)
    for r in range(args.reps):
        pick=idx[rng.choice(len(idx),size=min(npts,len(idx)),replace=False)]
        pr,pc=pick[:,0],pick[:,1]; pv=dtm[pr,pc]
        interp=griddata((pc,pr),pv,(XX,YY),method="linear")
        nz=griddata((pc,pr),pv,(XX,YY),method="nearest"); bad=~np.isfinite(interp); interp[bad]=nz[bad]
        m=valid; maes.append(float(np.nanmean(np.abs(interp-dtm)[m])))
    mae=np.mean(maes)
    print(f"  GEDI-density interpolation ({npts} pts ~ {args.gedi_per_km2:.0f}/km^2): "
          f"DTM MAE = {mae:.2f} m  (+/- {np.std(maes):.2f})")
    print(f"  => {'HEADROOM: interpolation fails here; optical could help' if mae>3 else 'NO headroom: interpolation suffices; fusion not worth it'}")
    np.savez(os.path.join(out,f"{args.name}_precond.npz"),dtm=dtm,relief=relief,slope_std=slope_std,mae=mae)

if __name__=="__main__": sys.exit(main())
