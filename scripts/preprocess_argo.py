# scripts/preprocess_argo.py
"""
Interpolate the downloaded ARGO gridded temperature to 15 standard depths.
Auto-detects variable and dimension names.
"""

import xarray as xr
import numpy as np
import os

ARGO_PATH = "data/raw/argo/argo_raw.nc"
OUT_PATH  = "data/processed/argo_2023_06_std_depths.nc"

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ------------------------------------------------------------
# Load and inspect
# ------------------------------------------------------------
if not os.path.exists(ARGO_PATH):
    print(f"❌ File not found: {ARGO_PATH}")
    print("Please download the ARGO file first.")
    exit(1)

ds = xr.open_dataset(ARGO_PATH)
print("=" * 60)
print("ARGO RAW DATASET")
print("=" * 60)
print(f"Dimensions: {dict(ds.sizes)}")
print(f"Variables:  {list(ds.data_vars)}")
print(f"Coords:     {list(ds.coords)}")
print()

# ------------------------------------------------------------
# Auto-detect temperature variable
# ------------------------------------------------------------
temp_var = None
for candidate in ['ocean_temperature', 'TEMP', 'T_ANALYZED', 'temp',
                  'temperature', 'thetao', 'T']:
    if candidate in ds.data_vars:
        temp_var = candidate
        break

if temp_var is None:
    print(f"❌ No temperature variable found. Available: {list(ds.data_vars)}")
    exit(1)
print(f"✅ Temperature variable detected: {temp_var}")

# ------------------------------------------------------------
# Auto-detect depth dimension
# ------------------------------------------------------------
depth_dim = None
for candidate in ['mean_pressure', 'pressure', 'ZAX', 'depth',
                  'deptht', 'lev', 'pres']:
    if candidate in ds.dims or candidate in ds.coords:
        depth_dim = candidate
        break

if depth_dim is None:
    print(f"❌ No depth dimension found. Dims: {list(ds.dims)}")
    exit(1)
print(f"✅ Depth dimension detected: {depth_dim}")
print(f"   Original depth values (first 5): {ds[depth_dim].values[:5]}")
print(f"   Original depth values (last 5):  {ds[depth_dim].values[-5:]}")
print()

# ------------------------------------------------------------
# Interpolate to standard depths
# ------------------------------------------------------------
print(f"Interpolating to {len(STANDARD_DEPTHS)} standard depths...")
argo_std = ds.interp({depth_dim: STANDARD_DEPTHS}, method='linear')

# Rename depth dimension to 'depth' for consistency
if depth_dim != 'depth':
    argo_std = argo_std.rename({depth_dim: 'depth'})

# Rename temperature variable to 'temperature'
if temp_var != 'temperature':
    argo_std = argo_std.rename({temp_var: 'temperature'})

# ------------------------------------------------------------
# Standardize lat/lon names if needed
# ------------------------------------------------------------
rename_dict = {}
if 'latitude' in argo_std.coords or 'latitude' in argo_std.dims:
    rename_dict['latitude'] = 'lat'
if 'longitude' in argo_std.coords or 'longitude' in argo_std.dims:
    rename_dict['longitude'] = 'lon'
if rename_dict:
    argo_std = argo_std.rename(rename_dict)

# ------------------------------------------------------------
# Check for NaNs
# ------------------------------------------------------------
temp_data = argo_std['temperature'].values
n_nan = int(np.isnan(temp_data).sum())
n_total = temp_data.size
print(f"  NaNs after interpolation: {n_nan:,} / {n_total:,} ({100 * n_nan / n_total:.1f}%)")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
argo_std.to_netcdf(OUT_PATH)
print(f"\n✅ Saved to: {OUT_PATH}")
print(f"   File size:  {os.path.getsize(OUT_PATH) / 1e6:.2f} MB")
print(f"   Dimensions: {dict(argo_std.sizes)}")
print(f"   Variables:  {list(argo_std.data_vars)}")
print()
print("=" * 60)
print("Next step: python scripts/predict_for_argo.py")
print("=" * 60)