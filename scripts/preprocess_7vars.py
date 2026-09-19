# scripts/preprocess_7vars.py
"""
Preprocess 7-variable dataset (June 2023):
GLORYS, SST, SSS, SSH, OSCAR currents (U,V), CCMP winds (U,V)

Fixes applied:
- Longitude monotonic issue (per-file load + concat)
- June 2023 filter
- OSCAR dim/coord mismatch (.sel instead of .interp)
- CCMP 6-hourly → daily resampling
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
RAW_DIR = "data/raw"
OUT_PATH = "data/processed/oceanembed_7vars_june2023.nc"
os.makedirs("data/processed", exist_ok=True)

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
TARGET_RES = 0.25
INPUT_VARS = ['sst', 'sss', 'ssh', 'u_current', 'v_current', 'u_wind', 'v_wind']

TIME_START = "2023-06-01"
TIME_END = "2023-06-30"


# ============================================================
# HELPERS
# ============================================================
def convert_time(ds):
    """Convert cftime to standard datetime64 if needed."""
    if 'time' in ds.coords:
        first = ds.time.values[0]
        if hasattr(first, 'calendar'):
            new_time = pd.to_datetime([str(t) for t in ds.time.values])
            ds = ds.assign_coords(time=new_time)
    return ds


def rename_coords(ds):
    """Standardize coord/dim names to lat/lon."""
    swap = {}
    if 'latitude' in ds.dims and 'lat' in ds.coords:
        swap['latitude'] = 'lat'
    if 'longitude' in ds.dims and 'lon' in ds.coords:
        swap['longitude'] = 'lon'
    if swap:
        ds = ds.swap_dims(swap)

    rn = {}
    if 'latitude' in ds.dims and 'lat' not in ds.dims:
        rn['latitude'] = 'lat'
    if 'longitude' in ds.dims and 'lon' not in ds.dims:
        rn['longitude'] = 'lon'
    if rn:
        ds = ds.rename(rn)
    return ds


def normalize_lon(ds):
    """Convert lon from 0-360 to -180..180."""
    name = None
    for c in ['lon', 'longitude']:
        if c in ds.coords or c in ds.dims:
            name = c
            break
    if name is None:
        return ds
    if float(ds[name].max()) > 180:
        ds = ds.assign_coords({name: (((ds[name] + 180) % 360) - 180)})
    return ds.sortby(name)


def load_all(folder):
    """Load all .nc files, filter to June 2023, concatenate safely."""
    files = sorted(glob.glob(os.path.join(folder, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No files in {folder}")
    print(f"  Found {len(files)} files")

    datasets = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            ds = convert_time(ds)
            ds = rename_coords(ds)
            ds = normalize_lon(ds)

            if 'time' in ds.coords:
                ds = ds.sel(time=slice(TIME_START, TIME_END))

            if len(ds.time) > 0:
                datasets.append(ds)
        except Exception as e:
            print(f"    ⚠️ Skipping {os.path.basename(f)}: {str(e)[:80]}")

    print(f"  Loaded {len(datasets)} files with June 2023 data")

    if not datasets:
        raise ValueError(f"No June 2023 data in {folder}")

    ds = xr.concat(datasets, dim='time', data_vars='minimal', coords='minimal')
    ds = ds.sortby('time')

    # Deduplicate time
    _, idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=idx)

    return ds


# ============================================================
# STEP 1: GLORYS
# ============================================================
print("=" * 60)
print("STEP 1: GLORYS")
print("=" * 60)
glorys = load_all(f"{RAW_DIR}/glorys")
print(f"  Dims: {dict(glorys.sizes)}")
print(f"  Time: {str(glorys.time.values[0])[:10]} to {str(glorys.time.values[-1])[:10]}")

lat_min, lat_max = float(glorys.lat.min()), float(glorys.lat.max())
lon_min, lon_max = float(glorys.lon.min()), float(glorys.lon.max())
target_lat = np.arange(lat_min, lat_max + TARGET_RES/2, TARGET_RES)
target_lon = np.arange(lon_min, lon_max + TARGET_RES/2, TARGET_RES)
print(f"  Target grid: {len(target_lat)} × {len(target_lon)}")

glorys_r = glorys.interp(lat=target_lat, lon=target_lon, method='linear')
glorys_std = glorys_r.interp(
    depth=STANDARD_DEPTHS, method='linear',
    kwargs={'fill_value': 'extrapolate'}
)
temperature = glorys_std['thetao'].rename('temperature')
print(f"  Temperature: {temperature.shape}")


# ============================================================
# STEP 2: SST
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: SST")
print("=" * 60)
sst = load_all(f"{RAW_DIR}/sst")
sst_r = sst.interp(lat=target_lat, lon=target_lon, method='linear')
sst_var = sst_r['analysed_sst'].rename('sst')
print(f"  SST: {sst_var.shape}")


# ============================================================
# STEP 3: SSS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: SSS")
print("=" * 60)
sss = load_all(f"{RAW_DIR}/sss")
if 'depth' in sss.dims:
    sss = sss.squeeze('depth', drop=True)
sss_r = sss.interp(lat=target_lat, lon=target_lon, method='linear')
sss_var = sss_r['sos'].rename('sss')
print(f"  SSS: {sss_var.shape}")


# ============================================================
# STEP 4: SSH
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SSH")
print("=" * 60)
ssh = load_all(f"{RAW_DIR}/ssh")
ssh_r = ssh.interp(lat=target_lat, lon=target_lon, method='linear')
ssh_var = ssh_r['sla'].rename('ssh')
print(f"  SSH: {ssh_var.shape}")


# ============================================================
# STEP 5: OSCAR (FIXED — .sel instead of .interp)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: OSCAR Currents (FIXED)")
print("=" * 60)
oscar = load_all(f"{RAW_DIR}/oscar")
print(f"  OSCAR vars: {list(oscar.data_vars)}")
print(f"  OSCAR dims: {dict(oscar.sizes)}")

u_var_name = 'u' if 'u' in oscar.data_vars else 'uo' if 'uo' in oscar.data_vars else None
v_var_name = 'v' if 'v' in oscar.data_vars else 'vo' if 'vo' in oscar.data_vars else None

if u_var_name is None or v_var_name is None:
    print(f"❌ OSCAR u/v not found. Available: {list(oscar.data_vars)}")
    exit(1)

# Squeeze depth if exists
for d in ['depth', 'lev']:
    if d in oscar.dims:
        oscar = oscar.isel({d: 0}, drop=True)

# Sort coords
oscar = oscar.sortby('lat').sortby('lon')
print(f"  After sort — lat: {float(oscar.lat.min()):.2f} to {float(oscar.lat.max()):.2f}")
print(f"  After sort — lon: {float(oscar.lon.min()):.2f} to {float(oscar.lon.max()):.2f}")

# Use .sel with nearest (avoids dim/coord mismatch)
u_current = oscar[u_var_name].sel(
    lat=target_lat, lon=target_lon, method='nearest'
).rename('u_current')

v_current = oscar[v_var_name].sel(
    lat=target_lat, lon=target_lon, method='nearest'
).rename('v_current')

n_valid = int(np.sum(~np.isnan(u_current.values)))
n_total = u_current.values.size
print(f"  U_current: {u_current.shape}")
print(f"  Valid u: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")


# ============================================================
# STEP 6: CCMP WINDS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: CCMP Winds")
print("=" * 60)
ccmp = load_all(f"{RAW_DIR}/ccmp")
print(f"  CCMP vars: {list(ccmp.data_vars)}")
print(f"  CCMP dims: {dict(ccmp.sizes)}")

u_wind_name = 'uwnd' if 'uwnd' in ccmp.data_vars else 'u10' if 'u10' in ccmp.data_vars else None
v_wind_name = 'vwnd' if 'vwnd' in ccmp.data_vars else 'v10' if 'v10' in ccmp.data_vars else None

if u_wind_name is None or v_wind_name is None:
    print(f"❌ CCMP uwnd/vwnd not found. Available: {list(ccmp.data_vars)}")
    exit(1)

# Sort coords
ccmp = ccmp.sortby('lat').sortby('lon')

# Resample 6-hourly → daily if needed
if len(ccmp.time) > 60:
    print("  Resampling 6-hourly → daily...")
    ccmp = ccmp.resample(time='1D').mean()

# Use .sel with nearest
u_wind = ccmp[u_wind_name].sel(
    lat=target_lat, lon=target_lon, method='nearest'
).rename('u_wind')

v_wind = ccmp[v_wind_name].sel(
    lat=target_lat, lon=target_lon, method='nearest'
).rename('v_wind')

n_valid = int(np.sum(~np.isnan(u_wind.values)))
n_total = u_wind.values.size
print(f"  U_wind: {u_wind.shape}")
print(f"  Valid uwnd: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")


# ============================================================
# STEP 7: TIME ALIGNMENT
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Time alignment")
print("=" * 60)
common_time = temperature.time
print(f"  Reference: {str(common_time.values[0])[:10]} to {str(common_time.values[-1])[:10]}")


def align(da, target_time):
    """Align time axis using reindex with nearest (preserves values)."""
    return da.reindex(time=target_time, method='nearest')


inputs = {
    'sst':       align(sst_var, common_time),
    'sss':       align(sss_var, common_time),
    'ssh':       align(ssh_var, common_time),
    'u_current': align(u_current, common_time),
    'v_current': align(v_current, common_time),
    'u_wind':    align(u_wind, common_time),
    'v_wind':    align(v_wind, common_time),
}


# ============================================================
# STEP 8: MERGE (force temperature grid)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Merging (forced to GLORYS grid)")
print("=" * 60)

# Force every input to exactly match temperature's grid
temperature_grid = xr.Dataset({'temperature': temperature})

aligned_inputs = {}
for k, v in inputs.items():
    aligned_inputs[k] = v.reindex(
        lat=temperature.lat,
        lon=temperature.lon,
        time=temperature.time,
        method='nearest'
    )

final = xr.merge([aligned_inputs[k] for k in INPUT_VARS] + [temperature_grid])

print(f"  Final grid: {len(final.lat)} lat × {len(final.lon)} lon × {len(final.time)} time")


# ============================================================
# STEP 9: FILL NaN (per-channel surface mean, per-depth temperature mean)
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Filling NaN")
print("=" * 60)
# Create ocean mask from surface temperature before filling
ocean_mask = ~np.isnan(final['temperature'].values[0, 0])
final['ocean_mask'] = (('lat', 'lon'), ocean_mask.astype(np.float32))

for v in INPUT_VARS:
    arr = final[v].values
    n_nan = int(np.isnan(arr).sum())
    if n_nan > 0:
        mean_val = float(np.nanmean(arr))
        arr[np.isnan(arr)] = mean_val
        pct = 100 * n_nan / arr.size
        print(f"  {v}: filled {n_nan:,} NaN ({pct:.1f}%) with {mean_val:.3f}")
    final[v].values[:] = arr.astype(np.float32)

# Subsurface temperature: fill NaN PER DEPTH SLICE with ocean mean at that specific depth
temp_arr = final['temperature'].values  # (time, depth, lat, lon)
for d in range(temp_arr.shape[1]):
    slice_d = temp_arr[:, d]
    n_nan_d = int(np.isnan(slice_d).sum())
    if n_nan_d > 0:
        depth_mean = float(np.nanmean(slice_d))
        slice_d[np.isnan(slice_d)] = depth_mean
        print(f"  temperature depth {STANDARD_DEPTHS[d]:>4}m: filled {n_nan_d:,} NaN with depth mean {depth_mean:.2f} °C")
    temp_arr[:, d] = slice_d
final['temperature'].values[:] = temp_arr.astype(np.float32)


# ============================================================
# STEP 10: SAVE
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Saving")
print("=" * 60)
final.to_netcdf(OUT_PATH)
print(f"\n✅ Saved: {OUT_PATH}")
print(f"   Size: {os.path.getsize(OUT_PATH)/1e6:.1f} MB")
print(f"   Dims: {dict(final.sizes)}")
print(f"   Variables: {list(final.data_vars)}")
print(f"   Total samples: {len(final.time) * len(final.lat) * len(final.lon):,}")