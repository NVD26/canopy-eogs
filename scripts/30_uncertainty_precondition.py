#!/usr/bin/env python3
"""
30_uncertainty_precondition.py — [U0] does an uncertainty signal predict the true DSM error?
Builds a per-pixel multi-view disagreement map (std across all per-view DSMs EOGS++ rendered),
plus a DSM-roughness baseline, and correlates each with |DSM - lidar| (Spearman + sparsification
error / AUSE). If disagreement is informative -> foundation for calibrated satellite-GS uncertainty.

  python scripts/30_uncertainty_precondition.py --exp ~/eogs-src/EOGS2/output/eogsplus_rpc_ba_JAX_068 \
      --scene JAX_068 --iters 14200 --eogs2-dir ~/eogs-src/EOGS2
"""
import os, sys, glob, argparse, numpy as np
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp",required=True)
    ap.add_argument("--scene",required=True)
    ap.add_argument("--iters",type=int,required=True)
    ap.add_argument("--eogs2-dir",default=os.path.expanduser("~/eogs-src/EOGS2"))
    a=ap.parse_args()
    import rasterio
    from rasterio.warp import reproject, Resampling
    from scipy.stats import spearmanr
    from scipy.ndimage import generic_filter, uniform_filter

    # ---- collect ALL per-view DSMs (train + test), exclude Nadir/msi ----
    pv=[]
    for split in ("train","test"):
        pv += glob.glob(os.path.join(a.exp,f"{split}_opNone/ours_{a.iters}/dsm/*"))
    pv=[p for p in pv if p.endswith(".iio") and "Nadir" not in os.path.basename(p) and "msi" not in os.path.basename(p)]
    print(f"[views] {len(pv)} per-view DSMs found")
    if len(pv)<3:
        dd=glob.glob(os.path.join(a.exp,"test_opNone",f"ours_{a.iters}","dsm","*"))
        print("  [diag] dsm dir contains:", [os.path.basename(x) for x in dd[:10]])
    if not pv: return 2

    # ---- GT grid + UTM crs (crs from a per-view DSM; grid from GT tif or .txt) ----
    with rasterio.open(pv[0]) as f: utm_crs=f.crs
    gt_tif=os.path.join(a.eogs2_dir,"data","Truth",f"{a.scene}_DSM.tif")
    with rasterio.open(gt_tif) as f:
        gt=f.read(1).astype(np.float64); H,W=gt.shape; gt_tr=f.transform; gt_has=f.crs is not None
    gt[gt<-1e3]=np.nan
    if not gt_has or gt_tr.a==1.0:
        txt=os.path.join(a.eogs2_dir,"data","Truth",f"{a.scene}_DSM.txt")
        roi=np.loadtxt(txt); xoff,yoff,size,res=float(roi[0]),float(roi[1]),int(roi[2]),float(roi[3])
        from rasterio.transform import from_origin
        gt_tr=from_origin(xoff, yoff+size*res, res, res)
    print(f"[grid] {W}x{H}  crs={utm_crs}  transform set")

    # ---- reproject each per-view DSM to the GT grid -> stack ----
    stack=np.full((len(pv),H,W),np.nan)
    for i,p in enumerate(pv):
        with rasterio.open(p) as src:
            dst=np.full((H,W),np.nan)
            reproject(rasterio.band(src,1),dst,src_transform=src.transform,src_crs=src.crs,
                      dst_transform=gt_tr,dst_crs=utm_crs,resampling=Resampling.bilinear)
            dst[dst<-1e3]=np.nan; stack[i]=dst
    nvalid=np.sum(np.isfinite(stack),axis=0)
    mean=np.nanmean(stack,axis=0); disagree=np.nanstd(stack,axis=0)

    # ---- prediction = multi-view mean, registered to GT (median offset) ----
    m=np.isfinite(mean)&np.isfinite(gt)&(nvalid>=2)
    offset=float(np.nanmedian((mean-gt)[m]))
    pred=mean-offset; err=np.abs(pred-gt)
    print(f"[reg] median offset {offset:+.2f} m; valid pixels {int(m.sum())}; overall MAE {np.nanmean(err[m]):.3f} m")

    # ---- roughness baseline (local std of prediction) ----
    pf=np.nan_to_num(pred, nan=float(np.nanmedian(pred[m])))
    rough=np.sqrt(np.clip(uniform_filter(pf**2,5)-uniform_filter(pf,5)**2,0,None))

    # ---- tree mask (optional) ----
    tm=os.path.join(a.eogs2_dir,"src","gaussiansplatting","scripts","eval","tree_masks",f"{a.scene}.png")
    tree=None
    if os.path.exists(tm):
        with rasterio.open(tm) as f: t=f.read(1)>0.5
        if t.shape==(H,W): tree=t

    def sparsification_auc(unc, e, mask):
        idx=np.where(mask.ravel())[0]; u=unc.ravel()[idx]; ee=e.ravel()[idx]
        order_u=idx[np.argsort(-u)]; order_o=idx[np.argsort(-ee)]   # desc
        fr=np.linspace(0,0.9,19); base=ee.mean()
        def curve(order):
            out=[]
            for f in fr:
                keep=order[int(f*len(order)):]      # remove top-f most-uncertain
                out.append(np.abs(e.ravel()[keep]).mean() if len(keep) else 0)
            return np.array(out)/base
        cu=curve(order_u); co=curve(order_o)        # method vs oracle (best)
        return float(np.trapz(cu-co,fr)/0.9), cu, co
    def report(name,unc,mask):
        u=unc.ravel()[mask.ravel()]; e=err.ravel()[mask.ravel()]
        ok=np.isfinite(u)&np.isfinite(e)
        rho=spearmanr(u[ok],e[ok]).correlation
        ause,_,_=sparsification_auc(unc,err,mask&np.isfinite(unc))
        print(f"  {name:28s} Spearman rho={rho:+.3f}   AUSE={ause:.3f}  (rho>0 & low AUSE = informative)")
        return rho,ause

    print("\n================ U0: does uncertainty predict error? ================")
    print(f"  (n per-view DSMs = {len(pv)})")
    print("  --- ALL valid pixels ---")
    report("multi-view disagreement", disagree, m)
    report("DSM roughness (baseline)", rough, m)
    if tree is not None:
        print("  --- VEGETATION pixels ---");  report("disagreement (veg)", disagree, m&tree); report("roughness (veg)", rough, m&tree)
        print("  --- BUILDING pixels ---");     report("disagreement (bldg)", disagree, m&~tree); report("roughness (bldg)", rough, m&~tree)
    print("=====================================================================")
    out=os.path.join(a.exp,"u0_uncertainty.npz")
    np.savez(out,disagree=disagree,rough=rough,err=err,pred=pred,gt=gt,nvalid=nvalid)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(15,4.5))
        def sh(x,im,t,**k): c=x.imshow(im,**k);x.set_title(t);plt.colorbar(c,ax=x,fraction=.046)
        vm=float(np.nanpercentile(err[m],95))
        sh(ax[0],np.where(m,err,np.nan),"|DSM - lidar| error",vmin=0,vmax=vm)
        sh(ax[1],np.where(m,disagree,np.nan),"multi-view disagreement (uncertainty)",vmin=0,vmax=float(np.nanpercentile(disagree[m],95)))
        ax[2].scatter(disagree[m],err[m],s=1,alpha=.1); ax[2].set_xlabel("disagreement"); ax[2].set_ylabel("|error|"); ax[2].set_title("uncertainty vs error")
        plt.tight_layout(); fig.savefig(os.path.join(a.exp,"u0_uncertainty.png"),dpi=110)
        print("figure ->",os.path.join(a.exp,"u0_uncertainty.png"))
    except Exception as e: print("figure skipped:",e)
if __name__=="__main__": sys.exit(main())
