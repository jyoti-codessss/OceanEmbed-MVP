# scripts/predict_full.py
"""
Run the trained hybrid model on the full dataset using patch extraction.
"""
import xarray as xr
import numpy as np
import torch
import os, sys, warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from hybrid_model import OceanEmbedHybrid

DATA_PATH  = "data/processed/oceanembed_7vars_june2023.nc"
MODEL_PATH = "models/hybrid_best.pt"
STATS_PATH = "models/hybrid_normalization_stats.npz"
OUT_PATH   = "data/processed/oceanembed_2023_06_predictions.nc"

INPUT_VARS = ['sst', 'sss', 'ssh', 'u_current', 'v_current', 'u_wind', 'v_wind']
N_INPUTS   = 7
N_DEPTHS   = 15
PATCH_SIZE = 8
HALF       = PATCH_SIZE // 2
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
ds = xr.open_dataset(DATA_PATH).load()
stats = np.load(STATS_PATH)
in_mean, in_std = stats['input_mean'], stats['input_std']
tgt_mean, tgt_std = stats['target_mean'], stats['target_std']

model = OceanEmbedHybrid(in_channels=N_INPUTS, patch_size=PATCH_SIZE,
                         n_depths=N_DEPTHS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

inputs = np.stack([ds[v].values for v in INPUT_VARS], axis=1)
for c in range(N_INPUTS):
    ch = inputs[:, c]
    ch[np.isnan(ch)] = np.nanmean(ch)
inputs_n = (inputs - in_mean) / in_std

T, _, H, W = inputs_n.shape
padded = np.pad(inputs_n, ((0,0),(0,0),(HALF,HALF),(HALF,HALF)), mode='edge')

print(f"Extracting patches for {T} days x {H}x{W} grid...")
patches = np.zeros((T, H, W, N_INPUTS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
for t in range(T):
    for i in range(H):
        for j in range(W):
            patches[t, i, j] = padded[t, :, i:i+PATCH_SIZE, j:j+PATCH_SIZE]

X = patches.reshape(-1, N_INPUTS, PATCH_SIZE, PATCH_SIZE)
print(f"  Total patches: {X.shape[0]:,}")

preds_l = []
bs = 512
with torch.no_grad():
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEVICE)
        preds_l.append(model(xb).cpu().numpy())

preds_n = np.concatenate(preds_l)
preds_real = preds_n * tgt_std.squeeze() + tgt_mean.squeeze()
preds_3d = preds_real.reshape(T, H, W, N_DEPTHS).transpose(0, 3, 1, 2).astype(np.float32)

# Mask land points using ocean_mask if present
if 'ocean_mask' in ds:
    ocean_mask = ds['ocean_mask'].values > 0.5
    for d in range(N_DEPTHS):
        preds_3d[:, d, ~ocean_mask] = np.nan

out = xr.Dataset(
    {'temperature_reconstructed': (('time','depth','lat','lon'), preds_3d)},
    coords={'time': ds.time, 'depth': ds.depth, 'lat': ds.lat, 'lon': ds.lon}
)
out['temperature_reconstructed'].attrs['long_name'] = 'Reconstructed subsurface temperature'
out['temperature_reconstructed'].attrs['units'] = 'degrees_C'
out.attrs['model'] = 'hybrid_cnn_vit_physics'
out.to_netcdf(OUT_PATH)
print(f"✅ Saved: {OUT_PATH}")
print(f"   Size: {os.path.getsize(OUT_PATH)/1e6:.2f} MB")