# scripts/inspect_data.py
# Inspect the structure of all downloaded 2023 datasets

import xarray as xr
import os

raw_dir = "data/raw"
subfolders = ["glorys", "sst", "sss", "ssh", "oscar", "ccmp"]

for sub in subfolders:
    sub_path = os.path.join(raw_dir, sub)
    if not os.path.exists(sub_path):
        print(f"[MISSING] Folder not found: {sub_path}")
        continue
    
    print("=" * 70)
    print(f"SUBFOLDER: {sub}")
    print("=" * 70)
    
    files_found = False
    for file in os.listdir(sub_path):
        if file.endswith((".nc", ".nc4")):
            files_found = True
            filepath = os.path.join(sub_path, file)
            print(f"\nFile: {file}")
            try:
                ds = xr.open_dataset(filepath)
                print(f"  Variables: {list(ds.data_vars)}")
                print(f"  Dimensions: {dict(ds.dims)}")
                print(f"  Coords: {list(ds.coords)}")
                if 'time' in ds.coords:
                    print(f"  Time range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
                if 'depth' in ds.dims:
                    d = ds.depth.values
                    print(f"  Depth levels ({len(d)}): {d[:3]}...{d[-3:]}")
                if 'lat' in ds.coords:
                    print(f"  Lat range: {float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f}")
                if 'lon' in ds.coords:
                    print(f"  Lon range: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}")
                ds.close()
            except Exception as e:
                print(f"  ERROR opening file: {e}")
    
    if not files_found:
        print(f"  No .nc or .nc4 files found in {sub_path}")

print("\n" + "=" * 70)
print("Inspection complete.")
print("=" * 70)