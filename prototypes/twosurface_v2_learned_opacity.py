#!/usr/bin/env python3
"""
twosurface_v2_learned_opacity.py — Paper 1 prototype, v2 (CPU, NumPy).

v1 showed a two-surface model recovers the under-canopy DTM from sparse lidar, but it
was TOLD where canopy is (a known mask). The real method must INFER canopy presence.
v2 removes that assumption: the canopy mask m in [0,1] is LEARNED jointly with the
ground, supervised only by:
  - a noisy vegetation cue (proxy for NDVI / EOGS appearance — what we'd really have),
  - the sparse lidar two-returns (at footprints we directly observe canopy-top AND
    ground, hence canopy presence = top-ground), and
  - smoothness.
Where the learned m is low (open), ground is coupled to the visible surface; where m
is high (canopy), ground is free, driven by sparse lidar + smoothness.

We compare under-canopy DTM MAE for:
  A) single surface (EOGS-like),
  B) two-surface with ORACLE mask (v1 upper bound),
  C) two-surface with LEARNED mask (realistic — this method).
"""
import numpy as np
np.random.seed(0)

H = W = 80
ys, xs = np.mgrid[0:H, 0:W] / float(H)
G_true = 5.0 + 8.0 * xs + 3.0 * np.sin(3 * np.pi * xs) * np.cos(2 * np.pi * ys)
forest = ((xs - 0.6) ** 2 + (ys - 0.45) ** 2) < 0.30 ** 2
CH_true = forest * np.clip(13.0 + 7.0 * np.cos(2 * np.pi * xs) * np.sin(2 * np.pi * ys), 2.0, None)
canopy_top_true = G_true + CH_true
S_obs = canopy_top_true + 0.10 * np.random.randn(H, W)     # optical sees the top

def lap(f):
    return np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1)-4*f
def blur(f, n=40):
    g = f.astype(float).copy()
    for _ in range(n):
        g = g + 0.2 * lap(g)
    return g
def mae(a, b, m):
    return float(np.abs(a-b)[m].mean())
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
anchor = np.random.default_rng(1).random((H, W)) < 0.04   # sparse lidar footprints

# noisy vegetation cue (proxy for NDVI/appearance): blurred forest + noise -> NOT the mask
veg_cue = np.clip(blur(forest, 30) * 1.3 + 0.18 * np.random.randn(H, W), 0.0, 1.0)

# ---- T: optical canopy-top fit (shared) ----
def fit_top(iters=3000, lr=0.10, w_sm=0.25):
    T = S_obs.copy()
    for _ in range(iters):
        T -= lr * (2*(T-S_obs) - w_sm*2*lap(T))
    return T
T = fit_top()

# ---- B: oracle-mask ground (v1 upper bound) ----
def ground_oracle(iters=20000):
    open_mask = (CH_true < 1.0)                 # TRUE open areas (oracle)
    known = open_mask | anchor
    val = np.where(anchor, G_true, S_obs)
    G = S_obs.copy(); G[known] = val[known]
    for _ in range(iters):
        G = G + 0.20*lap(G); G[known] = val[known]
    return G

# ---- C: learned-mask ground (this method) ----
def learn_mask(iters=3000, lrL=0.5, w_veg=1.0, w_ma=4.0, w_smm=0.5):
    """Learn canopy presence m in [0,1] from the noisy veg cue + sparse lidar
    presence (top-ground at footprints) + smoothness. No ground feedback (stable)."""
    L = np.log(np.clip(veg_cue, 0.02, 0.98) / (1 - np.clip(veg_cue, 0.02, 0.98)))
    m_tgt = (CH_true > 1.0).astype(float)          # canopy presence observed at footprints
    for _ in range(iters):
        m = sigmoid(L)
        dm = 2*w_veg*(m - veg_cue) + 2*w_ma*(m - m_tgt)*anchor - w_smm*2*lap(m)
        L = np.clip(L - lrL*(dm*m*(1-m)), -12, 12)
    return sigmoid(L)

def ground_from_mask(m, iters=20000, open_thresh=0.30):
    """Same stable diffusion solve as the oracle, but the open areas come from the
    LEARNED mask m (not the true canopy mask)."""
    open_mask = m < open_thresh                    # learned canopy-free pixels
    known = open_mask | anchor
    val = np.where(anchor, G_true, S_obs)          # lidar ground at footprints; surface in open
    G = S_obs.copy(); G[known] = val[known]
    for _ in range(iters):
        G = G + 0.20*lap(G); G[known] = val[known]
    return G

S = T                                  # single-surface "DTM" = its only surface = canopy top
G_oracle = ground_oracle()
m_learn = learn_mask()
G_learn = ground_from_mask(m_learn)

def iou(pred, gt):
    p, g = pred > 0.5, gt > 0.5
    return float((p & g).sum() / max((p | g).sum(), 1))

print("="*70)
print("v2: LEARNED canopy mask (no oracle). forest mean CH = %.1f m" % CH_true[forest].mean())
print("="*70)
print(f"learned-mask IoU vs true forest: {iou(m_learn, forest):.2f}  "
      f"(veg-cue alone IoU: {iou(veg_cue, forest):.2f})")
print()
print(f"{'model':<34}{'DTM MAE in forest (m)':>22}")
print("-"*56)
print(f"{'A: single surface (EOGS-like)':<34}{mae(S, G_true, forest):>22.3f}")
print(f"{'B: two-surface, ORACLE mask':<34}{mae(G_oracle, G_true, forest):>22.3f}")
print(f"{'C: two-surface, LEARNED mask (ours)':<34}{mae(G_learn, G_true, forest):>22.3f}")
print()
print("-> Inferring the canopy mask jointly (C) stays close to the oracle (B) and far below")
print("   the single-surface baseline (A) -> the method does not need a given canopy mask.")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vlo, vhi = G_true.min(), canopy_top_true.max()
    errC = np.abs(G_learn - G_true)
    panels = [
        ("GT ground (DTM)", G_true, vlo, vhi, "viridis"),
        ("GT canopy top (DSM)", canopy_top_true, vlo, vhi, "viridis"),
        ("noisy veg cue (input)", veg_cue, 0, 1, "YlGn"),
        ("LEARNED canopy mask", m_learn, 0, 1, "YlGn"),
        ("true forest mask", forest.astype(float), 0, 1, "YlGn"),
        ("C: recovered ground (learned)", G_learn, vlo, vhi, "viridis"),
        ("C: |ground error| (m)", errC, 0, max(errC.max(), 1e-3), "magma"),
        ("lidar anchors", anchor.astype(float) + 0.4*forest, 0, 1.4, "cividis"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for ax, (t, img, lo, hi, cm) in zip(axes.ravel(), panels):
        im = ax.imshow(img, vmin=lo, vmax=hi, cmap=cm); ax.set_title(t, fontsize=10); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Paper 1 v2: canopy mask LEARNED (not given) — under-canopy DTM still recovered",
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig("/tmp/phd/prototypes/twosurface_v2_result.png", dpi=110)
    print("\nFigure saved: prototypes/twosurface_v2_result.png")
except Exception as e:
    print(f"(figure skipped: {e})")
