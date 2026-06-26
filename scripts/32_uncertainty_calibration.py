#!/usr/bin/env python3
"""
32_uncertainty_calibration.py — [U1] build a CALIBRATED per-pixel DSM uncertainty for satellite GS
from signals already on disk (multi-view disagreement, DSM roughness, view-count), combine them, and
evaluate with LEAVE-ONE-SCENE-OUT calibration: fit on N-1 scenes, test on the held-out scene.
Reports, per held-out scene, for each signal and the COMBINED estimator:
  - Spearman rho (does it rank error correctly)
  - AUSE (sparsification: does removing high-uncertainty pixels cut error toward the oracle)
  - ECE in meters (after calibration: does predicted sigma match actual |error|)
Headline: the COMBINED, calibrated estimator should beat any single signal (esp. the trivial roughness)
and be well-calibrated -> first calibrated DSM uncertainty for satellite GS.

  python scripts/32_uncertainty_calibration.py --auto            # discover canonical eogsplus runs
  python scripts/32_uncertainty_calibration.py --exps DIR1 DIR2  # explicit exp dirs
"""
import os, sys, glob, argparse, numpy as np

def scene_signals(exp, scene, iters, eogs2_dir):
    import rasterio
    from rasterio.warp import reproject, Resampling
    from scipy.ndimage import uniform_filter
    pv=[]
    for sp in ("train","test"):
        pv+=glob.glob(os.path.join(exp,f"{sp}_opNone/ours_{iters}/dsm/*"))
    pv=[p for p in pv if p.endswith(".iio") and "Nadir" not in os.path.basename(p) and "msi" not in os.path.basename(p)]
    if len(pv)<3: return None
    with rasterio.open(pv[0]) as f: utm=f.crs
    gt_tif=os.path.join(eogs2_dir,"data","Truth",f"{scene}_DSM.tif")
    with rasterio.open(gt_tif) as f:
        gt=f.read(1).astype(np.float64); H,W=gt.shape; tr=f.transform; has=f.crs is not None
    gt[gt<-1e3]=np.nan
    if not has or tr.a==1.0:
        roi=np.loadtxt(os.path.join(eogs2_dir,"data","Truth",f"{scene}_DSM.txt"))
        from rasterio.transform import from_origin
        tr=from_origin(float(roi[0]),float(roi[1])+int(roi[2])*float(roi[3]),float(roi[3]),float(roi[3]))
    stack=np.full((len(pv),H,W),np.nan)
    for i,p in enumerate(pv):
        with rasterio.open(p) as src:
            d=np.full((H,W),np.nan)
            reproject(rasterio.band(src,1),d,src_transform=src.transform,src_crs=src.crs,
                      dst_transform=tr,dst_crs=utm,resampling=Resampling.bilinear)
            d[d<-1e3]=np.nan; stack[i]=d
    nvalid=np.sum(np.isfinite(stack),0); mean=np.nanmean(stack,0); disagree=np.nanstd(stack,0)
    m=np.isfinite(mean)&np.isfinite(gt)&(nvalid>=2)
    off=float(np.nanmedian((mean-gt)[m])); pred=mean-off; err=np.abs(pred-gt)
    pf=np.nan_to_num(pred,nan=float(np.nanmedian(pred[m])))
    rough=np.sqrt(np.clip(uniform_filter(pf**2,5)-uniform_filter(pf,5)**2,0,None))
    viewcount=nvalid.astype(np.float64)
    return dict(err=err,m=m,disagree=disagree,rough=rough,viewinv=1.0/np.clip(viewcount,1,None))

def ause(unc,err,m):
    idx=np.where(m.ravel())[0]; u=unc.ravel()[idx]; e=err.ravel()[idx]
    ok=np.isfinite(u)&np.isfinite(e); idx=idx[ok]; u=u[ok]; e=e[ok]
    ou=idx[np.argsort(-u)]; oo=idx[np.argsort(-e)]; fr=np.linspace(0,0.9,19); base=e.mean()
    cu=np.array([np.abs(err.ravel()[ou[int(f*len(ou)):]]).mean() if len(ou)-int(f*len(ou))>0 else 0 for f in fr])/base
    co=np.array([np.abs(err.ravel()[oo[int(f*len(oo)):]]).mean() if len(oo)-int(f*len(oo))>0 else 0 for f in fr])/base
    return float(np.trapz(cu-co,fr)/0.9)

def ece(sigma,err,m,nb=10):
    s=sigma.ravel()[m.ravel()]; e=err.ravel()[m.ravel()]; ok=np.isfinite(s)&np.isfinite(e); s=s[ok]; e=e[ok]
    qs=np.quantile(s,np.linspace(0,1,nb+1)); tot=0.0
    for i in range(nb):
        b=(s>=qs[i])&(s<=qs[i+1])
        if b.sum()>10: tot+=b.mean()*abs(s[b].mean()-e[b].mean())
    return float(tot)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exps",nargs="*",default=None)
    ap.add_argument("--auto",action="store_true")
    ap.add_argument("--eogs2-dir",default=os.path.expanduser("~/eogs-src/EOGS2"))
    a=ap.parse_args()
    from scipy.stats import spearmanr
    exps=a.exps or []
    if a.auto:
        for d in sorted(glob.glob(os.path.join(a.eogs2_dir,"output","eogsplus_rpcba_*_pan_3PAN"))): exps.append(d)
    scenes=[]
    for exp in exps:
        scene=os.path.basename(exp); import re
        mm=re.search(r'(JAX_\d+|IARPA_\d+)',scene); 
        if not mm: continue
        scene=mm.group(1)
        od=sorted(glob.glob(os.path.join(exp,"test_opNone","ours_*")))
        if not od: continue
        iters=os.path.basename(od[-1]).split("_")[-1]
        print(f"[load] {scene} iters {iters} ...",flush=True)
        s=scene_signals(exp,scene,iters,a.eogs2_dir)
        if s: s["scene"]=scene; scenes.append(s)
    if len(scenes)<2: print("!! need >=2 scenes for leave-one-out calibration; have",len(scenes)); return 1
    sig_names=["disagree","rough","viewinv"]
    print("\n================ U1: leave-one-scene-out uncertainty calibration ================")
    print(f"  {'held-out':10s} {'signal':12s} {'rho':>7s} {'AUSE':>7s} {'ECE(m)':>8s}")
    rows={}
    for ti,test in enumerate(scenes):
        tr=[s for j,s in enumerate(scenes) if j!=ti]
        m=test["m"]; err=test["err"]
        # single signals (quantile-calibrated to meters on train scenes)
        for name in sig_names:
            # calibrate: map signal->expected|err| via train-scene binning
            xs=np.concatenate([t[name].ravel()[t["m"].ravel()] for t in tr]); ys=np.concatenate([t["err"].ravel()[t["m"].ravel()] for t in tr])
            ok=np.isfinite(xs)&np.isfinite(ys); xs,ys=xs[ok],ys[ok]
            qs=np.quantile(xs,np.linspace(0,1,11)); cy=[ys[(xs>=qs[i])&(xs<=qs[i+1])].mean() if ((xs>=qs[i])&(xs<=qs[i+1])).sum()>10 else np.nan for i in range(10)]
            qc=0.5*(qs[:-1]+qs[1:]); cy=np.array(cy); good=np.isfinite(cy)
            sig=np.interp(test[name],qc[good],cy[good]) if good.sum()>1 else test[name]
            u=test[name].ravel()[m.ravel()]; e=err.ravel()[m.ravel()]; okk=np.isfinite(u)&np.isfinite(e)
            rho=spearmanr(u[okk],e[okk]).correlation
            print(f"  {test['scene']:10s} {name:12s} {rho:+7.3f} {ause(test[name],err,m):7.3f} {ece(sig,err,m):8.3f}")
        # COMBINED: linear regression |err| ~ signals (fit on train, predict on test)
        X=np.column_stack([np.concatenate([t[s].ravel()[t['m'].ravel()] for t in tr]) for s in sig_names])
        Y=np.concatenate([t["err"].ravel()[t['m'].ravel()] for t in tr])
        ok=np.isfinite(X).all(1)&np.isfinite(Y); X,Y=X[ok],Y[ok]
        from numpy.linalg import lstsq
        A=np.column_stack([X,np.ones(len(X))]); w,_,_,_=lstsq(A,Y,rcond=None)
        Xt=np.column_stack([test[s] for s in sig_names]); comb=(Xt.reshape(-1,len(sig_names))@w[:-1]+w[-1]).reshape(test["err"].shape)
        u=comb.ravel()[m.ravel()]; e=err.ravel()[m.ravel()]; okk=np.isfinite(u)&np.isfinite(e)
        rho=spearmanr(u[okk],e[okk]).correlation
        print(f"  {test['scene']:10s} {'COMBINED':12s} {rho:+7.3f} {ause(comb,err,m):7.3f} {ece(comb,err,m):8.3f}")
        print("  " + "-"*44)
    print("=================================================================================")
    print("  Headline check: does COMBINED beat single signals on AUSE & is ECE small (well-calibrated)?")
if __name__=="__main__": sys.exit(main())
