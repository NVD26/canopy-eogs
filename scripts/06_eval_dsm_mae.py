#!/usr/bin/env python3
"""
06_eval_dsm_mae.py — standalone DSM error metrics vs. ground-truth lidar DSM.

A backup / cross-check for the MAE that EOGS prints, and the basis for the
custom canopy-height eval in Paper 1. Computes MAE, median absolute error,
RMSE, and completeness (fraction of valid pixels within a height threshold),
with an optional median z-offset alignment (the DFC2019 metric tolerates a
global vertical shift).

Usage:
  python scripts/06_eval_dsm_mae.py --pred pred_DSM.tif --truth JAX_004_DSM.tif
  python scripts/06_eval_dsm_mae.py --pred p.tif --truth t.tif --no-align --threshold 1.0

Requires: rasterio, numpy.
"""
import argparse
import sys

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError:
    print("Missing rasterio. Install with: pip install rasterio")
    sys.exit(1)


def _read(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float64")
        nodata = ds.nodata
        profile = ds.profile
    return arr, nodata, profile


def _resample_pred_to_truth(pred_path, truth_path):
    """Reproject/resample the prediction onto the truth grid so pixels align."""
    with rasterio.open(truth_path) as t:
        dst = np.full((t.height, t.width), np.nan, dtype="float64")
        with rasterio.open(pred_path) as p:
            reproject(
                source=rasterio.band(p, 1),
                destination=dst,
                src_transform=p.transform,
                src_crs=p.crs,
                dst_transform=t.transform,
                dst_crs=t.crs,
                resampling=Resampling.bilinear,
            )
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predicted DSM GeoTIFF")
    ap.add_argument("--truth", required=True, help="ground-truth DSM GeoTIFF")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="completeness threshold in meters (default 1.0)")
    ap.add_argument("--no-align", action="store_true",
                    help="disable median z-offset alignment")
    args = ap.parse_args()

    truth, t_nodata, _ = _read(args.truth)
    pred, p_nodata, p_profile = _read(args.pred)

    # If grids differ in shape/geref, resample pred onto the truth grid.
    if pred.shape != truth.shape:
        print(f"pred {pred.shape} != truth {truth.shape}: resampling pred onto truth grid")
        try:
            pred = _resample_pred_to_truth(args.pred, args.truth)
        except Exception as e:  # noqa: BLE001
            print(f"!! resample failed ({e}). Are both DSMs georeferenced?")
            return 1

    # Build a valid mask (drop nodata + non-finite on either raster).
    valid = np.isfinite(truth) & np.isfinite(pred)
    for arr, nd in ((truth, t_nodata), (pred, p_nodata)):
        if nd is not None:
            valid &= arr != nd
    # Common DSM nodata sentinels.
    for sentinel in (-9999.0, -10000.0, 0.0):
        valid &= ~np.isclose(truth, sentinel)
    if valid.sum() == 0:
        print("!! No overlapping valid pixels. Check nodata / georeferencing.")
        return 1

    diff = pred[valid] - truth[valid]
    if not args.no_align:
        offset = float(np.median(diff))
        diff = diff - offset
        print(f"applied median z-offset alignment: {offset:+.3f} m")

    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    med = float(np.median(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    completeness = float(np.mean(abs_diff <= args.threshold))

    print("==================== DSM metrics ====================")
    print(f"valid pixels:        {int(valid.sum()):,}")
    print(f"MAE (m):             {mae:.3f}")
    print(f"median |err| (m):    {med:.3f}")
    print(f"RMSE (m):            {rmse:.3f}")
    print(f"completeness@{args.threshold:g}m:  {completeness*100:.1f}%")
    print("=====================================================")
    print("Copy MAE + train time into STATUS.md §6 (Results log).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
