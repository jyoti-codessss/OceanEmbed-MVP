# scripts/download_podaac.py
"""
Download OSCAR + CCMP in bulk using single command per dataset.
"""
import os
import subprocess

# Config
START = "2023-06-01T00:00:00Z"
END   = "2023-11-30T23:59:59Z"
BBOX  = "80,5,95,20"  # West, South, East, North

os.makedirs("data/raw/oscar", exist_ok=True)
os.makedirs("data/raw/ccmp", exist_ok=True)

# ============================================================
# OSCAR — Single command for whole range
# ============================================================
print("=" * 60)
print("Downloading OSCAR currents (full 6-month range)")
print("=" * 60)

cmd = [
    "podaac-data-downloader",
    "-c", "OSCAR_L4_OC_FINAL_V2.0",
    "-d", "data/raw/oscar",
    "-sd", START,
    "-ed", END,
    f"-b={BBOX}",
    "--verbose"
]

print("Command:", " ".join(cmd))
print()
try:
    subprocess.run(cmd, check=True)
    print("\n✅ OSCAR download complete")
except subprocess.CalledProcessError as e:
    print(f"\n❌ OSCAR failed: {e}")

# ============================================================
# CCMP — Single command for whole range
# ============================================================
print("\n" + "=" * 60)
print("Downloading CCMP winds (full 6-month range)")
print("=" * 60)

cmd = [
    "podaac-data-downloader",
    "-c", "CCMP_WINDS_10M6HR_L4_V3.1",
    "-d", "data/raw/ccmp",
    "-sd", START,
    "-ed", END,
    f"-b={BBOX}",
    "--verbose"
]

print("Command:", " ".join(cmd))
print()
try:
    subprocess.run(cmd, check=True)
    print("\n✅ CCMP download complete")
except subprocess.CalledProcessError as e:
    print(f"\n❌ CCMP failed: {e}")

print("\n" + "=" * 60)
print("Done. Check data/raw/oscar and data/raw/ccmp")
print("=" * 60)