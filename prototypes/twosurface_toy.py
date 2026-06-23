#!/usr/bin/env python3
"""
twosurface_toy.py — FEASIBILITY PROTOTYPE for Paper 1's core idea.

Question: with sparse spaceborne-lidar anchors, can a two-surface (canopy / ground)
model recover the under-canopy DTM where a single-surface (EOGS-like) model cannot?

Standalone synthetic experiment (pure NumPy, CPU, no EOGS/GPU). NOT the real method;
it isolates the geometry question that de-risks the build.

Faithful observation model:
  - An optical satellite / EOGS sees the FIRST surface = the canopy TOP. The bare
    ground UNDER canopy is hidden from optical entirely.
  - GEDI full-waveform / ICESat-2 give, at sparse footprints, BOTH the canopy-top and
    the GROUND return -> the only signal that reveals under-canopy terrain.
  - A canopy-cover proxy (alpha, e.g. GEDI L2B cover or NDVI) tells us where the surface
    IS the ground (open areas) vs where ground must be inferred (under canopy).

Models (same data, same anchors):
  A) single surface S (EOGS-like): fits the optical top-surface signal. Its only
     elevation is S, so its implied DTM = S = canopy top under forest.
  B) two surfaces (ours): top T (fit to optical) + ground G (open-area surface where
     canopy is absent, sparse lidar ground returns under canopy, smoothness, G<=T).
     CHM = T - G.
"""
import numpy as np
np.random.seed(0)

H = W = 80
ys, xs = np.mgrid[0:H, 0:W] / float(H)

# ---- synthetic ground truth ----
G_true = 5.0 + 8.0 * xs + 3.0 * np.sin(3 * np.pi * xs) * np.cos(2 * np.pi * ys)   # smooth bare earth
forest = ((xs - 0.6) ** 2 + (ys - 0.45) ** 2) < 0.30 ** 2
CH_true = forest * np.clip(13.0 + 7.0 * np.cos(2 * np.pi * xs) * np.sin(2 * np.pi * ys), 2.0, None)
alpha = np.clip(forest * (0.55 + 0.30 * (0.5 + 0.5 * np.sin(4 * np.pi * xs) * np.cos(3 * np.pi * ys))), 0, 0.9)
canopy_top_true = G_true + CH_true
S_obs = canopy_top_true + 0.10 * np.random.randn(H, W)   # optical sees the TOP (small noise)

def lap(f):
    return np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f
def mae(a, b, m):
    return float(np.abs(a - b)[m].mean())
def make_anchors(frac, seed=1):
    return np.random.default_rng(seed).random((H, W)) < frac

# A) single surface: fit optical top + smoothness
def fit_single(iters=4000, lr=0.10, w_sm=0.25):
    S = S_obs.copy()
    for _ in range(iters):
        S -= lr * (2 * (S - S_obs) - w_sm * 2 * lap(S))   # sum-scaled data + smoothness
    return S

# B) two surfaces
def fit_two(anchor, iters=20000):
    """Top T fit to optical; ground G = stable smoothness diffusion pinned (Dirichlet)
    at known ground = open areas (visible surface) UNION sparse lidar ground anchors."""
    T = S_obs.copy()
    for _ in range(4000):
        T -= 0.10 * (2 * (T - S_obs) - 0.25 * 2 * lap(T))     # ||T-S_obs||^2 + smoothness
    open_mask = alpha < 0.10
    known = open_mask | anchor
    val = np.where(anchor, G_true, S_obs)                     # anchor=lidar ground; open=surface
    G = S_obs.copy(); G[known] = val[known]
    for _ in range(iters):
        G = G + 0.20 * lap(G)                                 # Jacobi diffusion (stable: 0.2*4<1)
        G[known] = val[known]                                 # re-pin known ground
    CH = np.clip(T - G, 0.0, None)
    return T, G, CH


print("=" * 72)
print("Two-surface feasibility prototype (synthetic, CPU)")
print(f"forest pixels {int(forest.sum())}/{H*W} | mean canopy height in forest "
      f"{CH_true[forest].mean():.1f} m")
print("=" * 72)

frac = 0.04
anchor = make_anchors(frac)
S = fit_single()
T, G, CH = fit_two(anchor)

print(f"\nAnchor density {frac*100:.0f}%  ({int((anchor&forest).sum())} ground anchors inside forest)\n")
print(f"{'model':<26}{'DSM/CHM-top MAE':>17}{'DTM MAE (m)':>14}")
print("-" * 57)
print(f"{'A: single surface (EOGS)':<26}{mae(S, canopy_top_true, forest):>17.3f}{mae(S, G_true, forest):>14.3f}")
print(f"{'B: two-surface (ours)':<26}{mae(T, canopy_top_true, forest):>17.3f}{mae(G, G_true, forest):>14.3f}")
print()
print(f"-> Both reconstruct the canopy-top surface equally well (no regression).")
print(f"-> Single surface has NO ground: its implied DTM is off by ~the canopy height "
      f"({mae(S, G_true, forest):.1f} m).")
print(f"-> Two-surface recovers the under-canopy ground to {mae(G, G_true, forest):.2f} m MAE "
      f"from {frac*100:.0f}% sparse anchors.\n")

print("Anchor-density sweep (two-surface under-canopy DTM MAE):")
print(f"{'density':>9}{'ground anchors/forest':>23}{'DTM MAE (m)':>14}")
for fr in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]:
    a = make_anchors(fr, seed=2) if fr > 0 else np.zeros((H, W), bool)
    _, Gh, _ = fit_two(a)
    print(f"{fr*100:>8.1f}%{int((a&forest).sum()):>23}{mae(Gh, G_true, forest):>14.3f}")
print("(density 0% = open-area + smoothness only, no under-canopy lidar -> ground unobserved)")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    errB = np.abs(G - G_true); errA = np.abs(S - G_true)
    vlo, vhi = G_true.min(), canopy_top_true.max()
    emax = max(errA.max(), 1e-3)
    panels = [
        ("GT ground (DTM)", G_true, vlo, vhi, "viridis"),
        ("GT canopy top (DSM)", canopy_top_true, vlo, vhi, "viridis"),
        ("B: recovered ground", G, vlo, vhi, "viridis"),
        ("B: recovered canopy top", T, vlo, vhi, "viridis"),
        ("A: single-surface 'DTM'", S, vlo, vhi, "viridis"),
        ("A: |DTM error| (m)", errA, 0, emax, "magma"),
        ("B: |DTM error| (m)", errB, 0, emax, "magma"),
        ("lidar ground anchors", anchor.astype(float) + 0.4 * forest, 0, 1.4, "cividis"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for ax, (t, img, lo, hi, cm) in zip(axes.ravel(), panels):
        im = ax.imshow(img, vmin=lo, vmax=hi, cmap=cm); ax.set_title(t, fontsize=10); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Paper 1 feasibility: sparse lidar + two surfaces recovers under-canopy DTM; "
                 "single surface cannot", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("/tmp/phd/prototypes/twosurface_toy_result.png", dpi=110)
    print("\nFigure saved: prototypes/twosurface_toy_result.png")
except Exception as e:
    print(f"(figure skipped: {e})")
