# scripts/fix_correlation.py
"""
Fix inflated correlation values in metrics JSON files.
Replaces flattened 0.998 correlation with honest mean per-depth correlation.
"""

import json
import numpy as np
import os

MODEL_DIR = "models"

# ============================================================
# FIX 1: hybrid_metrics.json (GLORYS)
# ============================================================
hybrid_path = os.path.join(MODEL_DIR, "hybrid_metrics.json")

with open(hybrid_path, "r") as f:
    hybrid_metrics = json.load(f)

# Compute mean per-depth correlation
per_depth_corr = hybrid_metrics['per_depth']['corr']
mean_corr_glorys = float(np.nanmean(per_depth_corr))

print("=" * 60)
print("FIXING: hybrid_metrics.json")
print("=" * 60)
print(f"  Old overall.corr: {hybrid_metrics['overall']['corr']:.4f} (flattened, inflated)")
print(f"  New overall.corr: {mean_corr_glorys:.4f} (mean per-depth)")

hybrid_metrics['overall']['corr'] = mean_corr_glorys
hybrid_metrics['overall']['corr_note'] = "Mean per-depth correlation (scientifically valid metric)"

with open(hybrid_path, "w") as f:
    json.dump(hybrid_metrics, f, indent=2)

print(f"  ✅ Saved: {hybrid_path}\n")

# ============================================================
# FIX 2: argo_validation_metrics.json (ARGO)
# ============================================================
argo_path = os.path.join(MODEL_DIR, "argo_validation_metrics.json")

with open(argo_path, "r") as f:
    argo_metrics = json.load(f)

per_depth_corr = argo_metrics['per_depth']['corr']
mean_corr_argo = float(np.nanmean(per_depth_corr))

print("=" * 60)
print("FIXING: argo_validation_metrics.json")
print("=" * 60)
print(f"  Old overall.corr: {argo_metrics['overall']['corr']:.4f} (flattened, inflated)")
print(f"  New overall.corr: {mean_corr_argo:.4f} (mean per-depth)")

argo_metrics['overall']['corr'] = mean_corr_argo
argo_metrics['overall']['corr_note'] = "Mean per-depth correlation (scientifically valid metric)"

with open(argo_path, "w") as f:
    json.dump(argo_metrics, f, indent=2)

print(f"  ✅ Saved: {argo_path}\n")

print("=" * 60)
print("✅ BOTH METRICS FILES FIXED")
print("=" * 60)
print(f"\nGLORYS mean correlation: {mean_corr_glorys:.3f}")
print(f"ARGO mean correlation:   {mean_corr_argo:.3f}")
print("\nNext step: Restart Streamlit dashboard")
print("   streamlit run scripts/app.py")