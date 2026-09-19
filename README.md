# OceanEmbed-MVP

**Subsurface Ocean Temperature Reconstruction from Satellite Surface Observations**

> SIH 2026 Problem Statement #01 — INCOIS (Indian National Centre for Ocean Information Services)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oceanembed-mvp.streamlit.app)

---

## 🌊 Problem

Reconstructing subsurface ocean temperature profiles from satellite-observable surface variables is critical for understanding ocean dynamics, predicting cyclone intensification, and monitoring marine ecosystems. Traditional methods rely on sparse in-situ observations (ARGO floats, ship-based measurements), leaving vast ocean regions unobserved.

## 🎯 Solution

**OceanEmbed** uses a hybrid CNN + Vision Transformer architecture with physics-informed constraints to reconstruct 3D subsurface temperature fields from 7 satellite surface variables:

| # | Variable | Source | Resolution |
|---|----------|--------|------------|
| 1 | Sea Surface Temperature (SST) | GLORYS12v1 | 0.083° → 0.25° |
| 2 | Sea Surface Salinity (SSS) | GLORYS12v1 | 0.083° → 0.25° |
| 3 | Sea Surface Height (SSH) | GLORYS12v1 | 0.083° → 0.25° |
| 4 | Zonal Current (U) | OSCAR | 0.25° |
| 5 | Meridional Current (V) | OSCAR | 0.25° |
| 6 | Zonal Wind (U) | CCMP | 0.25° |
| 7 | Meridional Wind (V) | CCMP | 0.25° |

**Output**: Temperature at **15 standard depth levels** (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m)

---

## 🏗️ Architecture

```
7 Surface Variables (8×8 patch)
        │
        ├──► CNN Branch (center-preserving spatial features)
        ├──► ViT Branch (multi-head self-attention)
        └──► Center Skip (direct surface-to-depth projection)
                │
                ▼
         Fusion (concat → 256-dim)
                │
                ▼
         Decoder → 15 Depth Temperatures
                │
                ▼
      Physics Loss (vertical thermal stability)
```

- **Parameters**: 342,863
- **Training**: Physics-informed MSE loss with vertical stability constraint
- **Input patch**: 8×8 grid cells (~200 km × 200 km)

---

## 📊 Results (Bay of Bengal, June 2023)

### GLORYS Test Set (held-out 5 days)
| Metric | Value |
|--------|-------|
| **RMSE** | 0.549 °C |
| **Bias** | 0.067 °C |
| **Correlation** | 0.815 |

### Independent ARGO Validation (2,710 float observations)
| Metric | Value |
|--------|-------|
| **RMSE** | 0.672 °C |
| **Correlation** | 0.775 |

### vs WOA23 Climatology Baseline
| Metric | Climatology | OceanEmbed | Improvement |
|--------|-------------|------------|-------------|
| RMSE | 0.691 °C | 0.671 °C | **+2.9%** |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ~2 GB disk space

### Installation

```bash
git clone https://github.com/jyoti-codessss/OceanEmbed-MVP.git
cd OceanEmbed-MVP
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run scripts/app.py
```

### Full Pipeline (from scratch)

```bash
# 1. Download satellite data (requires Copernicus Marine credentials)
python scripts/download_copernicus.py

# 2. Preprocess and merge 7 variables
python scripts/preprocess_7vars.py

# 3. Train the hybrid model
python scripts/train_hybrid.py

# 4. Generate full predictions
python scripts/predict_full.py

# 5. Validate against ARGO floats
python scripts/validate_argo.py

# 6. Compare with WOA23 climatology
python scripts/climatology_baseline.py

# 7. Launch interactive dashboard
streamlit run scripts/app.py
```

---

## 📁 Project Structure

```
OceanEmbed-MVP/
├── scripts/
│   ├── app.py                    # Streamlit dashboard
│   ├── preprocess_7vars.py       # Data preprocessing pipeline
│   ├── hybrid_model.py           # Model architecture (CNN + ViT + Skip)
│   ├── train_hybrid.py           # Training with physics loss
│   ├── predict_full.py           # Full-domain prediction
│   ├── validate_argo.py          # Independent ARGO validation
│   ├── climatology_baseline.py   # WOA23 baseline comparison
│   ├── download_copernicus.py    # Copernicus data download
│   ├── download_argo.py          # ARGO data download
│   └── download_podaac.py        # PODAAC data download
├── models/
│   ├── hybrid_best.pt            # Trained model weights
│   ├── hybrid_metrics.json       # Training & test metrics
│   ├── hybrid_normalization_stats.npz
│   ├── argo_validation_metrics.json
│   └── climatology_baseline.json
├── data/
│   └── processed/
│       ├── oceanembed_7vars_june2023.nc    # Preprocessed input data
│       ├── oceanembed_2023_06_predictions.nc  # Model predictions
│       └── argo_2023_06_std_depths.nc      # ARGO observations
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Technical Details

### Physics-Informed Loss
The model uses a custom physics loss that penalizes gravitational instability — enforcing that temperature should generally decrease with depth (buoyant stability), while preserving thermocline structure:

```
L_total = L_MSE + λ · L_physics
L_physics = mean(ReLU(ΔT - 0.1°C)²)  for adjacent depth pairs
```

### Data Pipeline
- **Source**: Copernicus Marine Service (GLORYS12v1, CCMP, OSCAR)
- **Region**: Bay of Bengal (5°N–20°N, 78°E–93°E)
- **Period**: June 2023 (30 daily snapshots)
- **Resolution**: 0.25° × 0.25° spatial, daily temporal
- **Ocean masking**: Land points filtered using NaN-count threshold

### Validation Strategy
1. **Train/Val/Test split**: 20/5/5 days (temporal, not random)
2. **GLORYS test**: Evaluate on 5 held-out days model never saw
3. **ARGO validation**: Compare against independent float observations
4. **Climatology baseline**: Compare against WOA23 monthly climatology

---

## 📜 License

This project is part of the Smart India Hackathon (SIH) 2026 submission for INCOIS Problem Statement #01.

---

## 👥 Team

Built for SIH 2026 — Ministry of Education's Innovation Cell × AICTE × INCOIS
