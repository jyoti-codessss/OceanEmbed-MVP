# scripts/diagnose_oscar.py
import xarray as xr
import numpy as np
import glob

print("=" * 60)
print("OSCAR DIAGNOSTIC")
print("=" * 60)

files = sorted(glob.glob("data/raw/oscar/*.nc"))
print(f"Total files: {len(files)}")

ds = xr.open_dataset(files[0])
print(f"\nOriginal structure:")
print(f"  Dims: {dict(ds.sizes)}")
print(f"  Coords: {list(ds.coords)}")
print(f"  Data vars: {list(ds.data_vars)}")

print(f"\nLat:")
print(f"  Range: {float(ds.lat.min())} to {float(ds.lat.max())}")
print(f"  First 5: {ds.lat.values[:5]}")
print(f"  Last 5: {ds.lat.values[-5:]}")
print(f"  Monotonic: {bool(ds.lat.to_index().is_monotonic_increasing)}")

print(f"\nLon:")
print(f"  Range: {float(ds.lon.min())} to {float(ds.lon.max())}")
print(f"  First 5: {ds.lon.values[:5]}")
print(f"  Last 5: {ds.lon.values[-5:]}")
print(f"  Monotonic: {bool(ds.lon.to_index().is_monotonic_increasing)}")

# Check u variable
u = ds['u']
print(f"\nU variable:")
print(f"  Shape: {u.shape}")
print(f"  Non-NaN count: {int(np.sum(~np.isnan(u.values)))}")
print(f"  NaN count: {int(np.sum(np.isnan(u.values)))}")
print(f"  NaN %: {100 * np.isnan(u.values).sum() / u.values.size:.1f}%")

# Try direct nearest selection in Bay of Bengal
print(f"\nTesting nearest selection at (15N, 88E):")
try:
    val = u.sel(lat=15, lon=88, method='nearest')
    print(f"  Nearest value: {float(val.values[0]) if val.size else 'N/A'}")
    print(f"  Nearest lat: {float(val.lat.values)}")
    print(f"  Nearest lon: {float(val.lon.values)}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Try interp
print(f"\nTesting interp at (15N, 88E):")
try:
    val = u.interp(lat=15, lon=88)
    print(f"  Interp value: {float(val.values[0]) if val.size else 'N/A'}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Count valid u values in Bay of Bengal region
print(f"\nBay of Bengal region (5-20N, 80-95E):")
u_bob = u.sel(
    lat=slice(5, 20),
    lon=slice(80, 95)
)
print(f"  Subset shape: {u_bob.shape}")
print(f"  Non-NaN: {int(np.sum(~np.isnan(u_bob.values)))}")
print(f"  NaN: {int(np.sum(np.isnan(u_bob.values)))}")
print(f"  Valid %: {100 * np.sum(~np.isnan(u_bob.values)) / u_bob.values.size:.1f}%")

# Check a single time slice
print(f"\nSingle time slice (t=0):")
u_t0 = u.isel(time=0)
u_bob_t0 = u_t0.sel(lat=slice(5, 20), lon=slice(80, 95))
print(f"  Region shape: {u_bob_t0.shape}")
print(f"  Non-NaN: {int(np.sum(~np.isnan(u_bob_t0.values)))} / {u_bob_t0.size}")