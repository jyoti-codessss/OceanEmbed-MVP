# scripts/download_argo.py
"""
Download Gridded ARGO temperature data from NOAA PMEL ERDDAP.
Primary source: argo_rfromv23_temp (ARGO RFROM v2.3)
Fallback: NCEI Argo Global Data Repository
"""

import xarray as xr
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
LON_MIN, LON_MAX = 80, 95
LAT_MIN, LAT_MAX = 5, 20
TIME_START = "2023-06-01"
TIME_END   = "2023-06-30"

OUTPUT_DIR = "data/raw/argo"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "argo_raw.nc")

# ============================================================
# SOURCE 1: NOAA PMEL ERDDAP — ARGO RFROM v2.3
# ============================================================
print("=" * 70)
print("SOURCE 1: NOAA PMEL ERDDAP — ARGO RFROM v2.3 Temperature")
print("=" * 70)

url = "https://data.pmel.noaa.gov/pmel/erddap/griddap/argo_rfromv23_temp.nc"
print(f"URL: {url}")

try:
    ds = xr.open_dataset(url)
    print("✅ Connected!")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables:  {list(ds.data_vars)}")
    print(f"  Coords:     {list(ds.coords)}")

    # Auto-detect temperature variable
    temp_var = None
    for cand in ['temp', 'TEMP', 'temperature', 'T_ANALYZED', 'thetao']:
        if cand in ds.data_vars:
            temp_var = cand
            break

    if temp_var is None:
        print(f"  ⚠️ No temperature variable found. Available: {list(ds.data_vars)}")
        raise ValueError("Temperature variable not found")

    print(f"  ✅ Temperature variable: {temp_var}")

    # Subset to our region and time
    subset = ds
    if 'time' in ds.coords:
        subset = subset.sel(time=slice(TIME_START, TIME_END))
    for lat_name in ['latitude', 'lat']:
        if lat_name in ds.coords:
            subset = subset.sel({lat_name: slice(LAT_MIN, LAT_MAX)})
            break
    for lon_name in ['longitude', 'lon']:
        if lon_name in ds.coords:
            subset = subset.sel({lon_name: slice(LON_MIN, LON_MAX)})
            break

    if subset[temp_var].size == 0:
        print("  ⚠️ Subset is empty. Saving full dataset instead.")
        subset = ds

    subset.to_netcdf(OUTPUT_PATH)
    print(f"\n✅ SAVED to: {OUTPUT_PATH}")
    print(f"   File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.2f} MB")
    print(f"   Dims: {dict(subset.sizes)}")

    exit(0)

except Exception as e:
    print(f"❌ Failed: {str(e)[:200]}")

# ============================================================
# SOURCE 2: NCEI Argo Global Data Repository
# ============================================================
print("\n" + "=" * 70)
print("SOURCE 2: NCEI Argo Global Data Repository")
print("=" * 70)

# NCEI provides monthly Argo gridded data via direct download
# The June 2023 gridded product is available as a NetCDF file
NCEI_URL = (
    "https://www.ncei.noaa.gov/data/oceans/argo/gridded/"
    "rg_temp_sal/2023/202306.nc"
)
print(f"URL: {NCEI_URL}")

try:
    ds_ncei = xr.open_dataset(NCEI_URL)
    print("✅ Connected to NCEI!")
    print(f"  Dimensions: {dict(ds_ncei.sizes)}")
    print(f"  Variables:  {list(ds_ncei.data_vars)}")

    # Subset to our region
    subset = ds_ncei
    for lat_name in ['latitude', 'lat', 'LATITUDE']:
        if lat_name in ds_ncei.coords:
            subset = subset.sel({lat_name: slice(LAT_MIN, LAT_MAX)})
            break
    for lon_name in ['longitude', 'lon', 'LONGITUDE']:
        if lon_name in ds_ncei.coords:
            subset = subset.sel({lon_name: slice(LON_MIN, LON_MAX)})
            break

    subset.to_netcdf(OUTPUT_PATH)
    print(f"\n✅ SAVED to: {OUTPUT_PATH}")
    print(f"   Dims: {dict(subset.sizes)}")

    exit(0)

except Exception as e:
    print(f"❌ NCEI failed: {str(e)[:200]}")

# ============================================================
# ALL SOURCES FAILED
# ============================================================
print("\n" + "=" * 70)
print("⚠️ ALL AUTOMATIC SOURCES FAILED")
print("=" * 70)
print("""
Please download the ARGO data MANUALLY:

1. Go to: https://data.pmel.noaa.gov/pmel/erddap/griddap/argo_rfromv23_temp.html
2. Set the following:
   - Time: 2023-06-01 to 2023-06-30
   - Latitude: 5.0 to 20.0
   - Longitude: 80.0 to 95.0
   - Variable: temp (or T_ANALYZED)
3. Click "Submit" and download as NetCDF (.nc)
4. Save the file as: data/raw/argo/argo_raw.nc
5. Re-run: python scripts/preprocess_argo.py
""")