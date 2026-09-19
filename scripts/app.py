# scripts/app.py
import streamlit as st
import xarray as xr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os

st.set_page_config(
    page_title="OceanEmbed | Subsurface Temperature",
    page_icon="🌊", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%); }
    .hero {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 50%, #0284c7 100%);
        padding: 2rem 2.5rem; border-radius: 16px; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 10px 25px rgba(3,105,161,0.25);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .hero p { margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1.05rem; }
    .hero-tag {
        display: inline-block; background: rgba(255,255,255,0.15);
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.85rem; margin-right: 0.5rem;
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.25rem;
        border-left: 5px solid #0284c7; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-label {
        color: #64748b; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .metric-value { color: #0c4a6e; font-size: 1.8rem; font-weight: 700; margin-top: 0.25rem; }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #0c4a6e;
        margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.4rem;
        border-bottom: 2px solid #e0f2fe;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: #f1f5f9; padding: 6px; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 16px; font-weight: 600; color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background: white !important; color: #0c4a6e !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS & DATA  ← CHANGE 1: DATA_PATH updated
# ============================================================
DATA_PATH = "data/processed/oceanembed_7vars_june2023.nc"
PRED_PATH = "data/processed/oceanembed_2023_06_predictions.nc"
METRICS_PATH = "models/hybrid_metrics.json"
ARGO_METRICS_PATH = "models/argo_validation_metrics.json"
ARGO_MATCHED_PATH = "models/argo_matched.npz"

@st.cache_data
def load_all():
    ds = xr.open_dataset(DATA_PATH).load()
    ds_pred = xr.open_dataset(PRED_PATH).load()
    ds.close(); ds_pred.close()
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    ds['temperature_reconstructed'] = ds_pred['temperature_reconstructed']

    argo_metrics = None
    argo_matched = None
    if os.path.exists(ARGO_METRICS_PATH):
        with open(ARGO_METRICS_PATH) as f:
            argo_metrics = json.load(f)
    if os.path.exists(ARGO_MATCHED_PATH):
        argo_matched = dict(np.load(ARGO_MATCHED_PATH))

    return ds, metrics, argo_metrics, argo_matched

ds, metrics, argo_metrics, argo_matched = load_all()

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🌊 OceanEmbed</h1>
    <p>Satellite Embedding-Based Deep Learning for Subsurface Ocean Temperature Reconstruction</p>
    <div style="margin-top: 1rem;">
        <span class="hero-tag">📍 Bay of Bengal</span>
        <span class="hero-tag">📅 June 2023</span>
        <span class="hero-tag">🎯 0.25° Daily</span>
        <span class="hero-tag">🌡️ 15 Depths (0–1000 m)</span>
        <span class="hero-tag">🛰️ 7 Satellite Inputs</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR  ← CHANGE 2: Sidebar About updated
# ============================================================
st.sidebar.markdown("### 🎛️ Control Panel")

time_values = ds.time.values
dates_str = [str(t)[:10] for t in time_values]
selected_date_str = st.sidebar.selectbox("📅 **Select Date**", dates_str, index=0)
selected_date_idx = dates_str.index(selected_date_str)

variable = st.sidebar.radio(
    "📊 **Variable to Visualize**",
    ["Temperature (Reconstructed)", "Temperature (Ground Truth)",
     "SST (Input)", "SSS (Input)", "SSH (Input)",
     "Currents (U,V)", "Winds (U,V)"]
)

depth_levels = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
if "Temperature" in variable:
    selected_depth = st.sidebar.select_slider(
        "🌡️ **Depth (m)**", options=depth_levels, value=100
    )
else:
    selected_depth = None
    st.sidebar.selectbox("🌡️ Depth", ["N/A — Surface variable"], disabled=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.success(
    "**MVP Status:** Working prototype\n\n"
    "**Model:** Hybrid CNN + ViT + Physics\n\n"
    "**Inputs:** 7 satellite variables\n\n"
    "**Training:** 30 days (June 2023)"
)
st.sidebar.caption("🔬 Team T-Force | SIH 2026 | PS SIH26066")

# ============================================================
# METRIC CARDS
# ============================================================
st.markdown('<div class="section-title">📈 Model Performance — Held-out Test Set</div>',
            unsafe_allow_html=True)

o = metrics['overall']
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">{o['rmse']:.3f} <span style="font-size:1rem;">°C</span></div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #10b981;">
        <div class="metric-label">Bias</div>
        <div class="metric-value">{o['bias']:.3f} <span style="font-size:1rem;">°C</span></div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #8b5cf6;">
        <div class="metric-label">Correlation</div>
        <div class="metric-value">{o['corr']:.3f}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #f59e0b;">
        <div class="metric-label">Inputs</div>
        <div class="metric-value">7 <span style="font-size:0.85rem;">Variables</span></div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #06b6d4;">
        <div class="metric-label">Depths</div>
        <div class="metric-value">15 <span style="font-size:0.85rem;">Levels</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Spatial Map", "📉  Depth Profile", "📊  Metrics", "🎯  ARGO Validation"
])

# ------------------------------------------------------------
# TAB 1 — SPATIAL MAP  ← CHANGE 3: Currents + Winds added
# ------------------------------------------------------------
with tab1:
    if "Temperature" in variable:
        if "Reconstructed" in variable:
            field = ds['temperature_reconstructed'].sel(
                time=time_values[selected_date_idx], depth=selected_depth, method='nearest'
            ).values
            cbar = "Reconstructed T (°C)"
            title_icon, title_text = "🛠️", "Model Reconstruction"
        else:
            field = ds['temperature'].sel(
                time=time_values[selected_date_idx], depth=selected_depth, method='nearest'
            ).values
            cbar = "Ground Truth T (°C)"
            title_icon, title_text = "🎯", "GLORYS Ground Truth"
        colorscale = "RdYlBu_r"
    elif variable == "SST (Input)":
        field = ds['sst'].sel(time=time_values[selected_date_idx]).values
        cbar = "SST (°C)"; colorscale = "RdYlBu_r"
        title_icon, title_text = "🌡️", "Sea Surface Temperature"
    elif variable == "SSS (Input)":
        field = ds['sss'].sel(time=time_values[selected_date_idx]).values
        cbar = "SSS (psu)"; colorscale = "Viridis"
        title_icon, title_text = "🧂", "Sea Surface Salinity"
    elif variable == "SSH (Input)":
        field = ds['ssh'].sel(time=time_values[selected_date_idx]).values
        cbar = "SLA (m)"; colorscale = "RdBu_r"
        title_icon, title_text = "🌊", "Sea Level Anomaly"
    elif variable == "Currents (U,V)":
        u = ds['u_current'].sel(time=time_values[selected_date_idx]).values
        v = ds['v_current'].sel(time=time_values[selected_date_idx]).values
        field = np.sqrt(u**2 + v**2)
        cbar = "Current Speed (m/s)"; colorscale = "Viridis"
        title_icon, title_text = "🔄", "Surface Ocean Currents"
    else:  # Winds
        u = ds['u_wind'].sel(time=time_values[selected_date_idx]).values
        v = ds['v_wind'].sel(time=time_values[selected_date_idx]).values
        field = np.sqrt(u**2 + v**2)
        cbar = "Wind Speed (m/s)"; colorscale = "Plasma"
        title_icon, title_text = "💨", "Surface Winds"

    st.markdown(
        f'<div class="section-title">{title_icon} {title_text} — {selected_date_str}'
        + (f' @ {selected_depth}m' if selected_depth else '') + '</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure(data=go.Heatmap(
        z=field, x=ds.lon.values, y=ds.lat.values,
        colorscale=colorscale,
        colorbar=dict(title=cbar, thickness=15, len=0.8),
        hovertemplate="<b>Lat:</b> %{y:.2f}<br><b>Lon:</b> %{x:.2f}<br><b>Value:</b> %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(
        xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
        height=520, margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="#334155")
    )
    st.plotly_chart(fig, width='stretch')

    if variable == "Temperature (Reconstructed)":
        truth = ds['temperature'].sel(
            time=time_values[selected_date_idx], depth=selected_depth, method='nearest'
        ).values
        recon = ds['temperature_reconstructed'].sel(
            time=time_values[selected_date_idx], depth=selected_depth, method='nearest'
        ).values
        err = recon - truth

        st.markdown(
            f'<div class="section-title">⚠️ Error Map (Reconstructed − Truth) @ {selected_depth}m</div>',
            unsafe_allow_html=True
        )
        fig_err = go.Figure(data=go.Heatmap(
            z=err, x=ds.lon.values, y=ds.lat.values,
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="ΔT (°C)", thickness=15, len=0.8),
            hovertemplate="<b>ΔT:</b> %{z:.3f} °C<extra></extra>"
        ))
        fig_err.update_layout(
            xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
            height=420, margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Arial", size=12, color="#334155")
        )
        st.plotly_chart(fig_err, width='stretch')

        mean_abs = np.nanmean(np.abs(err))
        st.info(f"📌 **Mean |error| @ {selected_depth}m:** `{mean_abs:.3f} °C`")

# ------------------------------------------------------------
# TAB 2 — DEPTH PROFILE
# ------------------------------------------------------------
with tab2:
    st.markdown(
        '<div class="section-title">📉 Vertical Temperature Profile — Model vs Ground Truth</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        sel_lat = st.slider("📍 Latitude", float(ds.lat.min()), float(ds.lat.max()), 15.0, 0.25)
    with col_b:
        sel_lon = st.slider("📍 Longitude", float(ds.lon.min()), float(ds.lon.max()), 88.0, 0.25)

    truth = ds['temperature'].sel(
        time=time_values[selected_date_idx], lat=sel_lat, lon=sel_lon, method='nearest'
    ).values
    recon = ds['temperature_reconstructed'].sel(
        time=time_values[selected_date_idx], lat=sel_lat, lon=sel_lon, method='nearest'
    ).values

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=truth, y=depth_levels, mode='lines+markers',
        name='GLORYS (Truth)',
        line=dict(color='#0f172a', width=2.5, dash='dash'),
        marker=dict(size=9, color='#0f172a')
    ))
    fig2.add_trace(go.Scatter(
        x=recon, y=depth_levels, mode='lines+markers',
        name='OceanEmbed (Reconstructed)',
        line=dict(color='#dc2626', width=3.5),
        marker=dict(size=10, color='#dc2626')
    ))
    fig2.update_yaxes(autorange="reversed", title="Depth (m)", gridcolor="#e2e8f0")
    fig2.update_xaxes(title="Temperature (°C)", gridcolor="#e2e8f0")
    fig2.update_layout(
        height=560, template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(l=40, r=20, t=20, b=40),
        font=dict(family="Arial", size=12, color="#334155")
    )
    st.plotly_chart(fig2, width='stretch')

    rmse_pt = float(np.sqrt(np.mean((recon - truth) ** 2)))
    st.info(f"📌 **Profile RMSE @ ({sel_lat:.2f}°N, {sel_lon:.2f}°E):** `{rmse_pt:.3f} °C`")

# ------------------------------------------------------------
# TAB 3 — METRICS vs GLORYS
# ------------------------------------------------------------
with tab3:
    st.markdown(
        '<div class="section-title">📊 Depth-Stratified Skill Metrics (vs GLORYS Test Set)</div>',
        unsafe_allow_html=True
    )
    per = metrics['per_depth']

    fig3 = make_subplots(rows=1, cols=3,
                         subplot_titles=("RMSE (°C)", "Bias (°C)", "Correlation"))
    fig3.add_trace(go.Bar(
        y=per['depths'], x=per['rmse'], orientation='h',
        marker=dict(color='#dc2626', line=dict(width=0)),
        hovertemplate="Depth: %{y}m<br>RMSE: %{x:.3f} °C<extra></extra>"
    ), row=1, col=1)
    fig3.add_trace(go.Bar(
        y=per['depths'], x=per['bias'], orientation='h',
        marker=dict(color='#0284c7', line=dict(width=0)),
        hovertemplate="Depth: %{y}m<br>Bias: %{x:.3f} °C<extra></extra>"
    ), row=1, col=2)
    fig3.add_trace(go.Bar(
        y=per['depths'], x=per['corr'], orientation='h',
        marker=dict(color='#059669', line=dict(width=0)),
        hovertemplate="Depth: %{y}m<br>Corr: %{x:.3f}<extra></extra>"
    ), row=1, col=3)

    fig3.update_yaxes(autorange="reversed", title="Depth (m)", row=1, col=1, gridcolor="#e2e8f0")
    fig3.update_yaxes(row=1, col=2, gridcolor="#e2e8f0")
    fig3.update_yaxes(row=1, col=3, gridcolor="#e2e8f0")
    fig3.update_layout(
        height=620, showlegend=False, template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(family="Arial", size=11, color="#334155")
    )
    st.plotly_chart(fig3, width='stretch')

    st.markdown('<div class="section-title">📋 Full Metrics Table</div>', unsafe_allow_html=True)
    df = pd.DataFrame({
        'Depth (m)': per['depths'],
        'RMSE (°C)': np.round(per['rmse'], 3),
        'Bias (°C)': np.round(per['bias'], 3),
        'MAE (°C)': np.round(per['mae'], 3),
        'Correlation': np.round(per['corr'], 3)
    })
    st.dataframe(df, width='stretch', hide_index=True)

# ------------------------------------------------------------
# TAB 4 — ARGO VALIDATION
# ------------------------------------------------------------
with tab4:
    if argo_metrics is None:
        st.warning("ARGO validation metrics not found. Run `python scripts/validate_argo.py` first.")
    else:
        st.markdown(
            '<div class="section-title">🎯 Independent Validation Against Gridded ARGO Observations</div>',
            unsafe_allow_html=True
        )

        ao = argo_metrics['overall']
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc2626;">
                <div class="metric-label">ARGO RMSE</div>
                <div class="metric-value">{ao['rmse']:.3f} <span style="font-size:1rem;">°C</span></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <div class="metric-label">ARGO Bias</div>
                <div class="metric-value">{ao['bias']:.3f} <span style="font-size:1rem;">°C</span></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #8b5cf6;">
                <div class="metric-label">ARGO Corr</div>
                <div class="metric-value">{ao['corr']:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #10b981;">
                <div class="metric-label">Matched Points</div>
                <div class="metric-value">{ao['n_points']:,}</div>
            </div>""", unsafe_allow_html=True)

        st.info(
            f"📡 **Source:** {argo_metrics['source']}  |  "
            "**Note:** ARGO is an independent observational dataset — it was **not** used for training."
        )

        st.markdown('<div class="section-title">📊 ARGO Validation — Depth-Stratified Metrics</div>',
                    unsafe_allow_html=True)
        ap = argo_metrics['per_depth']

        fig4 = make_subplots(rows=1, cols=3,
                             subplot_titles=("RMSE (°C)", "Bias (°C)", "Correlation"))
        fig4.add_trace(go.Bar(
            y=ap['depths'], x=ap['rmse'], orientation='h',
            marker=dict(color='#dc2626'),
            hovertemplate="Depth: %{y}m<br>RMSE: %{x:.3f} °C<extra></extra>"
        ), row=1, col=1)
        fig4.add_trace(go.Bar(
            y=ap['depths'], x=ap['bias'], orientation='h',
            marker=dict(color='#f59e0b'),
            hovertemplate="Depth: %{y}m<br>Bias: %{x:.3f} °C<extra></extra>"
        ), row=1, col=2)
        fig4.add_trace(go.Bar(
            y=ap['depths'], x=ap['corr'], orientation='h',
            marker=dict(color='#059669'),
            hovertemplate="Depth: %{y}m<br>Corr: %{x:.3f}<extra></extra>"
        ), row=1, col=3)

        fig4.update_yaxes(autorange="reversed", title="Depth (m)", row=1, col=1, gridcolor="#e2e8f0")
        fig4.update_yaxes(row=1, col=2, gridcolor="#e2e8f0")
        fig4.update_yaxes(row=1, col=3, gridcolor="#e2e8f0")
        fig4.update_layout(
            height=620, showlegend=False, template="plotly_white",
            margin=dict(l=40, r=20, t=40, b=40),
            font=dict(family="Arial", size=11, color="#334155")
        )
        st.plotly_chart(fig4, width='stretch')

        st.markdown('<div class="section-title">📋 Full ARGO Validation Table</div>',
                    unsafe_allow_html=True)
        df_argo = pd.DataFrame({
            'Depth (m)': ap['depths'],
            'N Points': ap['n_points'],
            'RMSE (°C)': np.round(ap['rmse'], 3),
            'Bias (°C)': np.round(ap['bias'], 3),
            'MAE (°C)': np.round(ap['mae'], 3),
            'Correlation': np.round(ap['corr'], 3)
        })
        st.dataframe(df_argo, width='stretch', hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.9rem; padding:1rem 0;'>"
    "🌊 <b>OceanEmbed MVP</b> &nbsp;•&nbsp; Smart India Hackathon 2026 &nbsp;•&nbsp; "
    "Team T-Force &nbsp;•&nbsp; Problem Statement SIH26066<br>"
    "<span style='font-size:0.8rem;'>Powered by Physics-Informed Deep Learning | INCOIS | Ministry of Earth Sciences</span>"
    "</div>",
    unsafe_allow_html=True
)