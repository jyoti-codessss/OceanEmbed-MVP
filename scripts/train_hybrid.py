# scripts/train_hybrid.py
"""
Train OceanEmbed Hybrid Model (CNN + ViT + Physics Loss).
Uses 8x8 spatial patches from the 7 surface inputs with center-preserving skip connection.
"""
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from hybrid_model import OceanEmbedHybrid, physics_loss

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/processed/oceanembed_7vars_june2023.nc"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_VARS = ['sst', 'sss', 'ssh', 'u_current', 'v_current', 'u_wind', 'v_wind']
N_INPUTS  = 7
N_DEPTHS  = 15
PATCH_SIZE = 8
HALF = PATCH_SIZE // 2

TRAIN_DAYS = 20
VAL_DAYS   = 5
TEST_DAYS  = 5

BATCH_SIZE = 128
EPOCHS     = 80
LR         = 5e-4
LAMBDA_PHYS = 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTHS_TENSOR = torch.tensor(DEPTHS, dtype=torch.float32).to(DEVICE)

print(f"Device: {DEVICE}")

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

ds = xr.open_dataset(DATA_PATH).load()
inputs  = np.stack([ds[v].values for v in INPUT_VARS], axis=1)
targets = ds['temperature'].values
T, _, H, W = inputs.shape
print(f"  inputs:  {inputs.shape}")
print(f"  targets: {targets.shape}")

for c in range(N_INPUTS):
    ch = inputs[:, c]
    ch[np.isnan(ch)] = np.nanmean(ch)
for d in range(N_DEPTHS):
    l = targets[:, d]
    l[np.isnan(l)] = np.nanmean(l)

# Normalize
ti = inputs[:TRAIN_DAYS]
in_mean = ti.mean(axis=(0, 2, 3), keepdims=True)
in_std  = ti.std(axis=(0, 2, 3), keepdims=True) + 1e-8
inputs_n = (inputs - in_mean) / in_std

tt = targets[:TRAIN_DAYS]
tgt_mean = tt.mean(axis=(0, 2, 3), keepdims=True)
tgt_std  = tt.std(axis=(0, 2, 3), keepdims=True) + 1e-8
targets_n = (targets - tgt_mean) / tgt_std

np.savez(
    os.path.join(MODEL_DIR, "hybrid_normalization_stats.npz"),
    input_mean=in_mean, input_std=in_std,
    target_mean=tgt_mean, target_std=tgt_std,
    input_vars=np.array(INPUT_VARS)
)
print("  Saved normalization stats.")

# ============================================================
# PATCH EXTRACTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Building patch samples")
print("=" * 60)

ocean_mask = (ds['ocean_mask'].values > 0.5) if 'ocean_mask' in ds else np.ones((H, W), dtype=bool)

def extract_patches(inp, tgt, d0, d1):
    X_list, Y_list = [], []
    for t in range(d0, d1):
        img = inp[t]
        for i in range(HALF, H - HALF):
            for j in range(HALF, W - HALF):
                if ocean_mask[i, j]:
                    X_list.append(img[:, i-HALF:i+HALF, j-HALF:j+HALF])
                    Y_list.append(tgt[t, :, i, j])
    return np.stack(X_list).astype(np.float32), np.stack(Y_list).astype(np.float32)

X_train, Y_train = extract_patches(inputs_n, targets_n, 0, TRAIN_DAYS)
X_val, Y_val = extract_patches(inputs_n, targets_n, TRAIN_DAYS, TRAIN_DAYS + VAL_DAYS)
X_test, Y_test = extract_patches(inputs_n, targets_n,
                                  TRAIN_DAYS + VAL_DAYS,
                                  TRAIN_DAYS + VAL_DAYS + TEST_DAYS)

print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
    batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)),
    batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# BUILD MODEL (CNN + ViT only)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Building hybrid model (CNN + ViT)")
print("=" * 60)

model = OceanEmbedHybrid(in_channels=N_INPUTS, patch_size=PATCH_SIZE,
                         n_depths=N_DEPTHS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
n_cnn = sum(p.numel() for p in model.cnn.parameters())
n_vit = sum(p.numel() for p in model.vit.parameters())
print(f"  CNN:   {n_cnn:,}")
print(f"  ViT:   {n_vit:,}")
print(f"  Total: {n_params:,}")

# ============================================================
# TRAIN
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Training")
print("=" * 60)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6)

tgt_mean_t = torch.tensor(tgt_mean.squeeze(), dtype=torch.float32).to(DEVICE)
tgt_std_t  = torch.tensor(tgt_std.squeeze(), dtype=torch.float32).to(DEVICE)

best_val = float('inf')
best_epoch = 0
history = {'train': [], 'val': [], 'physics': []}
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    tl, pl = [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        mse = criterion(pred, yb)
        pred_real = pred * tgt_std_t + tgt_mean_t
        phys = physics_loss(pred_real, DEPTHS_TENSOR)
        loss = mse + LAMBDA_PHYS * phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl.append(mse.item()); pl.append(phys.item())

    model.eval()
    vl = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            vl.append(criterion(model(xb), yb).item())

    tr, va = float(np.mean(tl)), float(np.mean(vl))
    scheduler.step(epoch)
    history['train'].append(tr)
    history['val'].append(va)
    history['physics'].append(float(np.mean(pl)))

    if va < best_val:
        best_val = va
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "hybrid_best.pt"))

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {tr:.4f} | Val: {va:.4f} | "
              f"Phys: {np.mean(pl):.4f} | {time.time()-t0:.0f}s")

print(f"\n  Best epoch: {best_epoch+1} | Best val MSE: {best_val:.4f}")

# ============================================================
# TEST
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Testing")
print("=" * 60)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "hybrid_best.pt")))
model.eval()

preds_l, trues_l = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds_l.append(model(xb.to(DEVICE)).cpu().numpy())
        trues_l.append(yb.numpy())

preds_n = np.concatenate(preds_l)
true_n  = np.concatenate(trues_l)
pred_real = preds_n * tgt_std.squeeze() + tgt_mean.squeeze()
true_real = true_n  * tgt_std.squeeze() + tgt_mean.squeeze()

rmse = np.sqrt(np.mean((pred_real - true_real) ** 2, axis=0))
bias = np.mean(pred_real - true_real, axis=0)
mae  = np.mean(np.abs(pred_real - true_real), axis=0)
corr = np.array([
    np.corrcoef(pred_real[:, d], true_real[:, d])[0, 1]
    if true_real[:, d].std() > 1e-6 else np.nan
    for d in range(N_DEPTHS)
])

print(f"\n  {'Depth':>6} | {'RMSE':>7} | {'Bias':>7} | {'MAE':>7} | {'Corr':>6}")
print("  " + "-" * 50)
for i, d in enumerate(DEPTHS):
    print(f"  {d:>5}m | {rmse[i]:>7.3f} | {bias[i]:>7.3f} | "
          f"{mae[i]:>7.3f} | {corr[i]:>6.3f}")

overall = {
    'rmse': float(np.sqrt(np.mean((pred_real - true_real) ** 2))),
    'bias': float(np.mean(pred_real - true_real)),
    'corr': float(np.nanmean(corr)),
    'corr_note': "Mean per-depth correlation (scientifically valid metric)"
}
print(f"\n  OVERALL: RMSE={overall['rmse']:.3f} °C | "
      f"Bias={overall['bias']:.3f} °C | Corr={overall['corr']:.3f}")

# ============================================================
# SAVE
# ============================================================
metrics = {
    'overall': overall,
    'per_depth': {
        'depths': DEPTHS,
        'rmse': rmse.tolist(),
        'bias': bias.tolist(),
        'mae':  mae.tolist(),
        'corr': corr.tolist(),
    },
    'training_history': history,
    'best_epoch': best_epoch,
    'model_type': 'hybrid_cnn_vit_physics',
    'n_parameters': n_params
}
with open(os.path.join(MODEL_DIR, "hybrid_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

np.savez(os.path.join(MODEL_DIR, "hybrid_predictions.npz"),
         predictions=pred_real, targets=true_real, depths=np.array(DEPTHS))

print(f"\n✅ Hybrid training complete.")
print(f"   Model:   models/hybrid_best.pt")
print(f"   Metrics: models/hybrid_metrics.json")