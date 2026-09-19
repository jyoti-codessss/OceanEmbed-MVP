# scripts/climatology_baseline.py
"""
Compute climatology RMSE against the same ARGO points.
Compares against the model's ARGO RMSE on the same points.
"""
import xarray as xr
import numpy as np
import json
import os, sys
import glob

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIG
# ============================================================
ARGO_PATH = "data/processed/argo_2023_06_std_depths.nc"
WOA_DIR   = "data/raw/woa"
PRED_PATH = "data/processed/oceanembed_2023_06_predictions.nc"
OUT_JSON  = "models/climatology_baseline.json"

os.makedirs("models", exist_ok=True)
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


# ============================================================
# LOAD ARGO
# ============================================================
print("=" * 60)
print("Loading ARGO data...")
print("=" * 60)

argo = xr.open_dataset(ARGO_PATH).load()
argo_vals = argo['temperature'].values
print(f"  ARGO shape: {argo_vals.shape}")


# ============================================================
# LOAD WOA23
# ============================================================
print("\n" + "=" * 60)
print("Loading WOA23 climatology...")
print("=" * 60)

woa_files = glob.glob(os.path.join(WOA_DIR, "*.nc"))
if not woa_files:
    print(f"❌ No WOA file in {WOA_DIR}")
    exit(1)

print(f"  Using: {os.path.basename(woa_files[0])}")
woa = xr.open_dataset(woa_files[0], decode_times=False)
print(f"  WOA dims: {dict(woa.sizes)}")


# ============================================================
# TEMPERATURE VARIABLE
# ============================================================
temp_var = None
for cand in ['t_an', 't_mn', 'temperature', 'temp', 't_oa']:
    if cand in woa.data_vars:
        temp_var = cand
        break
if temp_var is None:
    temp_var = list(woa.data_vars)[0]
print(f"  Temperature variable: {temp_var}")


# ============================================================
# STANDARDIZE LON
# ============================================================
if 'lon' in woa.coords and float(woa.lon.max()) > 180:
    woa = woa.assign_coords(lon=(((woa.lon + 180) % 360) - 180)).sortby('lon')


# ============================================================
# DEPTH DIM
# ============================================================
depth_dim = None
for d in ['depth', 'lev', 'deptht']:
    if d in woa.dims:
        depth_dim = d
        break
if depth_dim is None:
    print(f"❌ No depth dim. Dims: {list(woa.dims)}")
    exit(1)

print(f"  Depth dim: {depth_dim}")
print(f"  Depth levels: {len(woa[depth_dim])}")


# ============================================================
# SQUEEZE EXTRA DIMS
# ============================================================
woa_temp = woa[temp_var]
for d in list(woa_temp.dims):
    if d not in [depth_dim, 'lat', 'lon']:
        woa_temp = woa_temp.isel({d: 0}, drop=True)


# ============================================================
# INTERPOLATE WOA TO ARGO GRID
# ============================================================
print("\n" + "=" * 60)
print("Interpolating WOA to ARGO grid...")
print("=" * 60)

argo_lat = argo['lat'].values
argo_lon = argo['lon'].values

woa_on_argo = woa_temp.interp(
    lat=argo_lat,
    lon=argo_lon,
    **{depth_dim: DEPTHS},
    method='linear'
)

woa_vals = np.squeeze(woa_on_argo.values)
while woa_vals.ndim > 3:
    woa_vals = woa_vals[0]

print(f"  WOA on ARGO grid: {woa_vals.shape}")

if woa_vals.shape != (15, 16, 16):
    print(f"❌ Unexpected shape: {woa_vals.shape}")
    exit(1)


# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("Loading model predictions...")
print("=" * 60)

pred_ds = xr.open_dataset(PRED_PATH).load()
pred_monthly = pred_ds['temperature_reconstructed'].mean(dim='time', keepdims=True)

pred_on_argo = pred_monthly.interp(
    lat=argo_lat, lon=argo_lon,
    depth=DEPTHS, method='linear'
).values[0]  # (15, 16, 16)

print(f"  Model on ARGO grid: {pred_on_argo.shape}")


# ============================================================
# COMPUTE RMSE ON SAME POINTS
# ============================================================
print("\n" + "=" * 60)
print("Computing RMSE on same points...")
print("=" * 60)

argo_2d = argo_vals[0]

# Combined mask: valid in ARGO, WOA, AND model
combined_mask = ~np.isnan(argo_2d) & ~np.isnan(woa_vals) & ~np.isnan(pred_on_argo)

n_valid = int(np.sum(combined_mask))
print(f"  Valid points (all three): {n_valid:,}")

# Flat arrays on common mask
a_flat = argo_2d[combined_mask]
m_flat = pred_on_argo[combined_mask]
c_flat = woa_vals[combined_mask]

# RMSE (overall)
model_rmse  = float(np.sqrt(np.mean((m_flat - a_flat) ** 2)))
clim_rmse   = float(np.sqrt(np.mean((c_flat - a_flat) ** 2)))
model_bias  = float(np.mean(m_flat - a_flat))
clim_bias   = float(np.mean(c_flat - a_flat))

# Improvement % (positive = model better than climatology)
improvement = (clim_rmse - model_rmse) / clim_rmse * 100

print(f"\n  Model RMSE:        {model_rmse:.3f} °C")
print(f"  Climatology RMSE:  {clim_rmse:.3f} °C")
print(f"  Improvement:       {improvement:.1f}%")


# ============================================================
# PER-DEPTH BREAKDOWN
# ============================================================
rmse_per_depth, bias_per_depth, n_per_depth = [], [], []
rmse_model_per_depth = []

for i in range(15):
    a = argo_2d[i].flatten()
    c = woa_vals[i].flatten()
    m = pred_on_argo[i].flatten()
    mask = ~np.isnan(a) & ~np.isnan(c) & ~np.isnan(m)
    a_v, c_v, m_v = a[mask], c[mask], m[mask]
    n_per_depth.append(len(a_v))

    if len(a_v) < 2:
        rmse_per_depth.append(np.nan)
        bias_per_depth.append(np.nan)
        rmse_model_per_depth.append(np.nan)
        continue

    rmse_per_depth.append(float(np.sqrt(np.mean((c_v - a_v) ** 2))))
    bias_per_depth.append(float(np.mean(c_v - a_v)))
    rmse_model_per_depth.append(float(np.sqrt(np.mean((m_v - a_v) ** 2))))


# ============================================================
# SAVE
# ============================================================
result = {
    'climatology_rmse': clim_rmse,
    'climatology_bias': clim_bias,
    'model_rmse': model_rmse,
    'model_bias': model_bias,
    'improvement_percent': improvement,
    'n_matched_points': n_valid,
    'per_depth': {
        'depths': DEPTHS,
        'climatology_rmse': rmse_per_depth,
        'climatology_bias': bias_per_depth,
        'model_rmse': rmse_model_per_depth,
        'n_points': n_per_depth
    },
    'source': 'WOA23 objectively analyzed climatology (June)'
}

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ BASELINE COMPLETE")
print(f"{'='*60}")
print(f"\n🎯 HEADLINE for Slide 2:")
print(f'   "Cuts subsurface temperature error by {improvement:.1f}% against')
print(f'    climatology, on {n_valid:,} independent ARGO observations')
print(f'    the model never saw."')
print(f"\n✅ Saved: {OUT_JSON}")