import os
import copernicusmarine

# Bay of Bengal region (same as June - Phase 1)
MIN_LON, MAX_LON = 80.0, 95.0
MIN_LAT, MAX_LAT = 5.0, 20.0

# INCOIS 15 Standard Depths
TARGET_DEPTHS = [0.494, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0,
                 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
MIN_DEPTH = 0.0
MAX_DEPTH = 1005.0

MONTHS_2023 = [
    ("2023-07-01", "2023-07-31", "07"),
    ("2023-08-01", "2023-08-31", "08"),
    ("2023-09-01", "2023-09-30", "09"),
    ("2023-10-01", "2023-10-31", "10"),
    ("2023-11-01", "2023-11-30", "11"),
]

DATASETS = {
    "glorys": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao"],
        "folder": "data/raw/glorys",
        "prefix": "glorys_2023_",
        "has_depth": True
    },
    "sst": {
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",  # ✅ FIXED (OSTIA)
        "variables": ["analysed_sst"],
        "folder": "data/raw/sst",
        "prefix": "sst_2023_",
        "has_depth": False
    },
    "sss": {
        "dataset_id": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",  # ✅ FIXED
        "variables": ["sos"],
        "folder": "data/raw/sss",
        "prefix": "sss_2023_",
        "has_depth": False
    },
    "ssh": {
    "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-demo-l4-duacs-0.125deg_P1D-i",
    "variables": ["sla"],
    "folder": "data/raw/ssh",
    "prefix": "ssh_2023_",
    "has_depth": False
}
}

for name, cfg in DATASETS.items():
    os.makedirs(cfg["folder"], exist_ok=True)
    print(f"\n--- Starting {name.upper()} downloads ---")

    for start_t, end_t, m_str in MONTHS_2023:
        out_file = f"{cfg['prefix']}{m_str}.nc"
        out_path = os.path.join(cfg["folder"], out_file)

        if os.path.exists(out_path):
            print(f"Skipping {out_file} (Already exists)")
            continue

        print(f"Downloading {out_file} ({start_t} to {end_t})...")

        kwargs = {
            "dataset_id": cfg["dataset_id"],
            "variables": cfg["variables"],
            "minimum_longitude": MIN_LON,
            "maximum_longitude": MAX_LON,
            "minimum_latitude": MIN_LAT,
            "maximum_latitude": MAX_LAT,
            "start_datetime": start_t,
            "end_datetime": end_t,
            "output_directory": cfg["folder"],
            "output_filename": out_file
        }

        if cfg["has_depth"]:
            kwargs["minimum_depth"] = MIN_DEPTH
            kwargs["maximum_depth"] = MAX_DEPTH

        try:
            copernicusmarine.subset(**kwargs)
        except Exception as e:
            print(f"Error downloading {out_file}: {e}")

print("\n✅ Download complete!")