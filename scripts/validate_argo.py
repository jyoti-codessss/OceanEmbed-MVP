# scripts/validate_argo.py
"""
Compare ARGO observations against model's reconstructed temperature.
Reports mean per-depth correlation (scientifically valid metric).
"""

import xarray as xr
import numpy as np
import pandas as pd
import json
import os, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# PATHS
# ============================================================
ARGO_PATH  = "data/processed/argo_2023_06_std_depths.nc"
PRED_PATH  = "data/processed/oceanembed_2023_06_predictions.nc"
OUT_DIR    = "models"
OUT_JSON   = os.path.join(OUT_DIR, "argo_validation_metrics.json")
os.makedirs(OUT_DIR, exist_ok=True)

DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ============================================================
# LOAD
# ============================================================
print("=" * 60)
print("Loading ARGO and model predictions")
print("=" * 60)

argo = xr.open_dataset(ARGO_PATH)
pred = xr.open_dataset(PRED_PATH)

print(f"ARGO dims:  {dict(argo.sizes)}")
print(f"Pred dims:  {dict(pred.sizes)}")

# ============================================================
# AVERAGE PREDICTIONS TO MONTHLY MEAN
# ============================================================
print("\nAveraging daily predictions to monthly mean...")
pred_monthly = pred['temperature_reconstructed'].mean(dim='time', keepdims=True)
print(f"  Monthly pred shape: {pred_monthly.shape}")

# ============================================================
# INTERPOLATE PRED TO ARGO GRID
# ============================================================
print("Interpolating model output to ARGO grid...")

argo_lat = argo['lat'].values
argo_lon = argo['lon'].values
argo_depths = argo['depth'].values

pred_on_argo = pred_monthly.interp(
    lat=argo_lat, lon=argo_lon,
    depth=argo_depths,
    method='linear'
).values

argo_vals = argo['temperature'].values

print(f"  ARGO shape:         {argo_vals.shape}")
print(f"  Model-on-ARGO shape: {pred_on_argo.shape}")

# ============================================================
# COMPUTE METRICS PER DEPTH
# ============================================================
print("\n" + "=" * 60)
print("Computing metrics vs ARGO")
print("=" * 60)

rmse_l, bias_l, mae_l, corr_l, n_l = [], [], [], [], []

for i, d in enumerate(DEPTHS):
    a = argo_vals[0, i, :, :].flatten()
    p = pred_on_argo[0, i, :, :].flatten()
    mask = ~np.isnan(a) & ~np.isnan(p)
    a = a[mask]
    p = p[mask]
    n_l.append(len(a))

    if len(a) < 2:
        rmse_l.append(np.nan); bias_l.append(np.nan)
        mae_l.append(np.nan);  corr_l.append(np.nan)
        continue

    rmse_l.append(float(np.sqrt(np.mean((p - a) ** 2))))
    bias_l.append(float(np.mean(p - a)))
    mae_l.append(float(np.mean(np.abs(p - a))))
    if a.std() > 1e-6:
        corr_l.append(float(np.corrcoef(p, a)[0, 1]))
    else:
        corr_l.append(np.nan)

# ============================================================
# OVERALL METRICS — FIXED: Mean per-depth correlation
# ============================================================
a_all = argo_vals.flatten()
p_all = pred_on_argo.flatten()
mask = ~np.isnan(a_all) & ~np.isnan(p_all)
a_all = a_all[mask]
p_all = p_all[mask]

mean_corr = float(np.nanmean(corr_l))  # ← FIXED: mean of per-depth correlations

overall = {
    'rmse': float(np.sqrt(np.mean((p_all - a_all) ** 2))),
    'bias': float(np.mean(p_all - a_all)),
    'mae':  float(np.mean(np.abs(p_all - a_all))),
    'corr': mean_corr,  # ← Honest metric
    'corr_note': 'Mean per-depth correlation (scientifically valid)',
    'n_points': int(len(a_all))
}

# ============================================================
# PRINT RESULTS
# ============================================================
print(f"\n  Valid matched points: {overall['n_points']:,}\n")
print(f"  {'Depth':>6} | {'N':>5} | {'RMSE':>7} | {'Bias':>7} | {'MAE':>7} | {'Corr':>6}")
print("  " + "-" * 50)
for i, d in enumerate(DEPTHS):
    n = n_l[i]
    if n < 2:
        print(f"  {d:>5}m | {n:>5} | {'—':>7} | {'—':>7} | {'—':>7} | {'—':>6}")
    else:
        print(f"  {d:>5}m | {n:>5} | {rmse_l[i]:>7.3f} | {bias_l[i]:>7.3f} | "
              f"{mae_l[i]:>7.3f} | {corr_l[i]:>6.3f}")

print(f"\n  ⭐ OVERALL vs ARGO:")
print(f"     RMSE:              {overall['rmse']:.3f} °C")
print(f"     Bias:              {overall['bias']:.3f} °C")
print(f"     MAE:               {overall['mae']:.3f} °C")
print(f"     Mean Correlation:  {overall['corr']:.3f} (per-depth mean)")

# ============================================================
# SAVE
# ============================================================
metrics = {
    'overall': overall,
    'per_depth': {
        'depths': DEPTHS,
        'rmse': rmse_l,
        'bias': bias_l,
        'mae':  mae_l,
        'corr': corr_l,
        'n_points': n_l
    },
    'source': 'NOAA PMEL ARGO RFROM 1x1 (independent)'
}

with open(OUT_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

np.savez(
    os.path.join(OUT_DIR, "argo_matched.npz"),
    argo=a_all, model=p_all
)

print(f"\n✅ Saved:")
print(f"   Metrics:  {OUT_JSON}")
print(f"   Matched:  {OUT_DIR}/argo_matched.npz")